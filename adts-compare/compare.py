import os
import pandas as pd
import shutil

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def compare_main():
    # Iterate through all environment variables, choose the best-performing model and write the pickle file to the working directory
    for k in os.environ.keys():
        if "PASSED_CONDITION" in k and os.environ[k] == "true":
            if not (
                os.path.exists("/cnvrg/model") or os.path.exists("/cnvrg/model.pkl")
            ):
                task_name = (
                    k.replace("CNVRG_", "").replace("_PASSED_CONDITION", "").lower()
                )
                if task_name == "autoencoder":
                    shutil.move("/input/" + task_name + "/model", cnvrg_workdir)
                    threshold = os.environ["CNVRG_" + task_name.upper() + "_THRESHOLD"]
                    df = pd.DataFrame(
                        {"winner": task_name, "threshold": threshold}, index=[0]
                    )
                    df.to_csv(cnvrg_workdir + "/winner_details.csv")
                else:
                    shutil.move("/input/" + task_name + "/model.pkl", cnvrg_workdir)
                print("Winner is: ", task_name)


if __name__ == "__main__":
    compare_main()
