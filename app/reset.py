import joblib, shutil

def reset_project():
    # Load your encoder
    labels = 0
    joblib.dump(labels, '/media/ammar/Partion2/Hazem Project/API/app/counter.joblib')


    shutil.copy("/media/ammar/Partion2/Hazem Project/encoder.joblib", "/media/ammar/Partion2/Hazem Project/API/app/encoder.joblib")
    shutil.copy("/media/ammar/Partion2/Hazem Project/IRISRecognizer.h5", "/media/ammar/Partion2/Hazem Project/API/app/IRISRecognizer.h5")

    return "Print reset project Successfully"

#reset_project()