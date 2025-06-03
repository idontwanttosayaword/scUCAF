from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from utils.sclayers import ZINBLoss, NBLoss
import math
import numpy as np
import pandas as pd  
from config import Config
from model.scUCAF import scUCAF
from dataset import CreateDataset
from utils.utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-c', '--config', type=str, default='configs\config.yaml',
                        help='Path to the config file')
    parser.add_argument('-r', '--run_dir', type=str, default=None,
                        help='Directory name under runs/ (default: None)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs for statistical evaluation')
    return parser.parse_args()


def perform_kmeans_clustering(z_numpy, y_true, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(z_numpy)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    centers = kmeans.cluster_centers_

    return nmi, ari, centers


def run_experiment(args, config, run_idx, num_runs):

    sc_dataset = CreateDataset(config)
    X1 = sc_dataset.adata1.X
    X_raw1 = sc_dataset.adata1.raw.X
    sf1 = sc_dataset.adata1.obs.size_factors
    X2 = sc_dataset.adata2.X
    X_raw2 = sc_dataset.adata2.raw.X
    sf2 = sc_dataset.adata2.obs.size_factors
    Y = sc_dataset.y

    num_genes = X1.shape[0]
    num_x2 = X2.shape[0]

    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))
    dataset = TensorDataset(
        torch.Tensor(X1),
        torch.Tensor(X_raw1),
        torch.Tensor(sf1),
        torch.Tensor(X2),
        torch.Tensor(X_raw2),
        torch.Tensor(sf2)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_cells = X1.shape[0]
    input_size1 = sc_dataset.adata1.n_vars
    input_size2 = sc_dataset.adata2.n_vars
    n_clusters = config.n_clusters if config.n_clusters else len(np.unique(Y))
    print(f"Run {run_idx + 1}/{num_runs} - n_clusters: {n_clusters}")

    net = scUCAF(num_cells, input_size1, input_size2,
                n_clusters,
                config.encoder1_layers, config.encoder2_layers, config.decoder1_layers, config.decoder2_layers)
    net = net.to(device)

    ZINB_loss = ZINBLoss()
    NB_loss = NBLoss()
    KL_loss = nn.KLDivLoss(size_average=False)

    base_run_root = Path('runs')
    if args.run_dir:
        run_root = base_run_root / args.run_dir
    else:
        run_root = base_run_root
    current_time = datetime.now()
    time_str = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    run_folder = run_root / f"{time_str}_run{run_idx + 1}"
    run_folder.mkdir(exist_ok=True, parents=True)

    print(f"\nRun {run_idx + 1}/{num_runs} - Model Structure:")
    print(str(net))
    net.train()

    if config.pre_train:
        print(f"Run {run_idx + 1}/{num_runs} - Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, amsgrad=True)

        for epoch in range(config.pre_train_epochs):
            epoch_loss = 0.0
            epoch_loss_x1 = 0.0
            epoch_loss_x2 = 0.0
            epoch_kl_loss_x1 = 0.0
            epoch_kl_loss_x2 = 0.0
            batch_count = 0

            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(
                    dataloader):
                optimizer.zero_grad()

                x1_tensor = x1_batch.to(device)
                x_raw1_tensor = x_raw1_batch.to(device)
                sf1_tensor = sf1_batch.to(device)
                x2_tensor = x2_batch.to(device)
                x_raw2_tensor = x_raw2_batch.to(device)
                sf2_tensor = sf2_batch.to(device)

                z_clean_fused, mean1, disp1, mean2, disp2, mu1, logvar1, mu2, logvar2 = net.forward_zinb(
                    x1_tensor, x2_tensor)

                loss_x1 = NB_loss(x_raw1_tensor, mean1, disp1, sf1_tensor)
                loss_x2 = NB_loss(x_raw2_tensor, mean2, disp2, sf2_tensor)

                kl_loss_x1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp()) / x1_tensor.size(0)
                kl_loss_x2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / x2_tensor.size(0)

                batch_loss = (loss_x1 + loss_x2) + 0.0001 * (kl_loss_x1 + kl_loss_x2)

                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count

            
            print(f'Run {run_idx + 1}/5 - Epoch [{epoch + 1}/{config.pre_train_epochs}] - '
                  f'Loss: {avg_loss:.4f} | ')

        
        torch.save(net.state_dict(), run_folder / "pre_train.pt")

    if not config.pre_train and config.weight_path:
        weight_path = config.weight_path
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint, strict=False)

    print(f"\nRun {run_idx + 1}/{num_runs} - Evaluating clustering:")
    z_numpy = net.encodeBatch(torch.tensor(X1).to(device),
                              torch.tensor(X2).to(device),
                              batch_size)
    z_data = torch.tensor(z_numpy).to(device)

    _, _, centers = perform_kmeans_clustering(z_numpy, Y, n_clusters)

    net.mu.data.copy_(torch.Tensor(centers).to(device))
    net.train()

    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, rho=.95)

    nmi = 0
    ari = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (x1_batch, x_raw1_batch, sf1_batch, x2_batch, x_raw2_batch, sf2_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            
            x1_tensor = x1_batch.to(device)
            x_raw1_tensor = x_raw1_batch.to(device)
            sf1_tensor = sf1_batch.to(device)
            x2_tensor = x2_batch.to(device)
            x_raw2_tensor = x_raw2_batch.to(device)
            sf2_tensor = sf2_batch.to(device)

            
            z_clean_fused, mean1, disp1, mean2, disp2, mu1, logvar1, mu2, logvar2, p1, p2, _, _ = net.forward(
                x1_tensor, x2_tensor)

            
            dist, _ = net.clustering_loss(z_clean_fused)
            y_pred = torch.argmin(dist, dim=1)
            sim_matrix = comprehensive_similarity(p1, p2)
            H, H_mat = high_confidence(z_clean_fused, net.mu, config.tao)
            M_mat = pseudo_matrix(y_pred, sim_matrix)
            neg_weights = torch.ones(p1.shape[0], p1.shape[0], device=device)
            neg_weights[H_mat] = M_mat[H_mat].data
            contrastive_loss = net.info_nce_loss(p1, p2, neg_weights)

            
            loss_x1 = NB_loss(x_raw1_tensor, mean1, disp1, sf1_tensor)
            loss_x2 = NB_loss(x_raw2_tensor, mean2, disp2, sf2_tensor)

            
            kl_loss_x1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp()) / x1_tensor.size(0)
            kl_loss_x2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / x2_tensor.size(0)

            _, cluster_loss = net.clustering_loss(z_clean_fused)

            batch_loss = (loss_x1 + loss_x2) + 0.0001 * (
                    kl_loss_x1 + kl_loss_x2) + contrastive_loss + cluster_loss

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        print(f'Run {run_idx + 1}/{num_runs} - Epoch [{epoch + 1}/{config.epochs}] - '
              f'Loss: {avg_loss:.4f} | ')

        z_numpy = net.encodeBatch(torch.tensor(X1, dtype=torch.float).to(device),
                                  torch.tensor(X2, dtype=torch.float).to(device),
                                  batch_size)
        z_data = torch.tensor(z_numpy, dtype=torch.float).to(device)

        dist, _ = net.clustering_loss(z_data)
        y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
        nmi = normalized_mutual_info_score(Y, y_pred)
        ari = adjusted_rand_score(Y, y_pred)

    return nmi, ari



if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)

    
    base_run_root = Path('runs')
    if args.run_dir:
        run_root = base_run_root / args.run_dir
    else:
        run_root = base_run_root
    run_root.mkdir(exist_ok=True, parents=True)

    
    nmi_results = []
    ari_results = []

    
    num_runs = args.runs
    for run_idx in range(num_runs):
        nmi, ari = run_experiment(args, config, run_idx, num_runs)
        nmi_results.append(nmi)
        ari_results.append(ari)

    
    nmi_mean = np.mean(nmi_results)
    nmi_std = np.std(nmi_results)
    ari_mean = np.mean(ari_results)
    ari_std = np.std(ari_results)

    
    print("\n===== Summary of 5 Runs =====")
    print(f"ENMI: {nmi_mean:.4f}±{nmi_std:.4f}")
    print(f"EARI: {ari_mean:.4f}±{ari_std:.4f}")

    with open(run_root / "summary_results.txt", "w") as f:
        f.write("===== Summary of Runs =====\n")
        f.write(f"Number of runs: {num_runs}\n\n")
        f.write(f"NMI: {nmi_mean:.4f}±{nmi_std:.4f}\n")
        f.write(f"ARI: {ari_mean:.4f}±{ari_std:.4f}\n")

    print(f"\nResults saved to {run_root}")