import os
import json
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dialign_python.dialign_python_offline import dialign

debate_path = "MAD_Debate_Process"

def plot_correlation(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    print(title)
    print(pearsonr(x, y))

def build_turns_from_debate(debate):
    """Returns a list[dict] with Speaker/Utterance rows."""
    data = []
    # seed row
    data.append({
        "Speaker": "Affirmative side",
        "Utterance": debate['players']['Affirmative side'][5]['content'],
    })
    # then alternate speakers over Negative side's utterances starting at index 2
    for i, utt in enumerate(debate['players']['Negative side'][2:]):
        speaker = "Negative side" if i % 2 == 0 else "Affirmative side"
        data.append({"Speaker": speaker, "Utterance": utt['content']})
    return data

def process_one_file(path_to_file):
    """
    Process a single debate file and return a dict of all metrics needed later.
    Uses a per-process temp CSV since `dialign` expects a CSV path.
    """
    try:
        with open(path_to_file, 'r') as f:
            debate = json.load(f)

        comet_score = debate['comet score']
        data = build_turns_from_debate(debate)

        # Write to a unique temp CSV for this process
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            pd.DataFrame(data).to_csv(tmp_path, index=False)

        try:
            speaker_independent, speaker_dependent, shared_expressions, self_repetitions, online_metrics = \
                dialign(tmp_path, "Speaker", "Utterance")
        finally:
            # Best-effort cleanup
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # Collect metrics we need for plotting
        return {
            "ok": True,
            "file": os.path.basename(path_to_file),
            "comet": comet_score,
            "er": speaker_independent['ER'],
            "ee": speaker_independent['EE'],
            "aff_rep": speaker_dependent['Affirmative side']['ER'],
            "neg_rep": speaker_dependent['Negative side']['ER'],
            "aff_est": speaker_dependent['Affirmative side']['EE'],
            "neg_est": speaker_dependent['Negative side']['EE'],
            "aff_init": speaker_dependent['Affirmative side']['Initiated'],
        }

    except Exception as e:
        # Return an error record instead of crashing the whole pool
        return {
            "ok": False,
            "file": os.path.basename(path_to_file),
            "error": f"{e.__class__.__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        }


def main():
    # Gather files
    files = [os.path.join(debate_path, f)
             for f in os.listdir(debate_path)
             if os.path.isfile(os.path.join(debate_path, f))]

    results = []
    # Use all CPUs by default; tweak max_workers if dialign is memory-hungry
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        futures = {ex.submit(process_one_file, p): p for p in files}
        for fut in as_completed(futures):
            res = fut.result()
            if res["ok"]:
                print("Processed:", res["file"])
            else:
                print("FAILED:", res["file"], "-", res["error"])
            results.append(res)

    # Filter out failures
    ok = [r for r in results if r["ok"]]

    # Build series for plotting (keeps each pair aligned)
    comet_scores = [r["comet"] for r in ok]
    er = [r["er"] for r in ok]
    ee = [r["ee"] for r in ok]
    affirmative_repetitions = [r["aff_rep"] for r in ok]
    negative_repetitions = [r["neg_rep"] for r in ok]
    affirmative_establishments = [r["aff_est"] for r in ok]
    negative_establishments = [r["neg_est"] for r in ok]
    affirmative_initiated = [r["aff_init"] for r in ok]

    # Plots (same as your original)
    plot_correlation(er, comet_scores, 'ER', 'COMET Score',
                     'Correlation between ER and COMET Score', 'er_comet_correlation.png')
    plot_correlation(ee, comet_scores, 'EE', 'COMET Score',
                     'Correlation between EE and COMET Score', 'ee_comet_correlation.png')
    plot_correlation(affirmative_repetitions, comet_scores, 'Affirmative Repetitions', 'COMET Score',
                     'Correlation between Affirmative Repetitions and COMET Score', 'affirmative_repetitions_comet_correlation.png')
    plot_correlation(negative_repetitions, comet_scores, 'Negative Repetitions', 'COMET Score',
                     'Correlation between Negative Repetitions and COMET Score', 'negative_repetitions_comet_correlation.png')
    plot_correlation(affirmative_establishments, comet_scores, 'Affirmative Establishments', 'COMET Score',
                     'Correlation between Affirmative Establishments and COMET Score', 'affirmative_establishments_comet_correlation.png')
    plot_correlation(negative_establishments, comet_scores, 'Negative Establishments', 'COMET Score',
                     'Correlation between Negative Establishments and COMET Score', 'negative_establishments_comet_correlation.png')
    plot_correlation(affirmative_initiated, comet_scores, 'Affirmative Initiated', 'COMET Score',
                     'Correlation between Affirmative Initiated and COMET Score', 'affirmative_initiated_comet_correlation.png')

if __name__ == "__main__":
    # Required on Windows/macOS for multiprocessing
    main()