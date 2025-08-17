def convert_results_to_csv(results):
    """
    :param results: list of dict, each dict has keys:
      epoch, global_acc, target_recall, asr, fp, fn, fp_rate, fn_rate, benign_killed, comm_cost_bytes, epoch_time_sec
    """
    if not results:
        return []

    fieldnames = [
        "epoch","global_acc","target_recall","asr",
        "fp","fn","fp_rate","fn_rate","benign_killed",
        "comm_cost_bytes","epoch_time_sec"
    ]
    cleaned_epoch_test_set_results = []
    for row in results:
        cleaned_epoch_test_set_results.append([row.get(k, "") for k in fieldnames])

    return cleaned_epoch_test_set_results
