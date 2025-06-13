import joblib, shutil, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def reset_project():
    labels = 0
    joblib.dump(labels, './counter.joblib')


    shutil.copy("../encoder.joblib", "./encoder.joblib")
    shutil.copy("../IRISRecognizer.h5", "./IRISRecognizer.h5")

    return "Print reset project Successfully"