from sklearn.metrics import r2_score
import numpy as np

def calculate_metric(preds, trues):
    error = trues - preds
    # 计算R²
    r2 = r2_score(trues, preds)  

    mean = np.mean(error)
    rmse = np.sqrt(np.sum(error**2)/len(trues))
    mae = np.sum(np.abs(error))/len(trues)

    real_mask = (1 - (trues == 0))
    loss = np.abs(error/(np.abs(trues)+1e-7))
    loss *= real_mask
    non_zero_len = np.sum(real_mask)
    if non_zero_len == 0:
        mape = 0
    else:
        mape = np.sum(loss)/non_zero_len

    return r2, mean, rmse, mae, mape
