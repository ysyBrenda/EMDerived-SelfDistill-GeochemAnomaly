# Detection anomaly scores：
#  data prepare：
#     ./data/rediction.pkl，
#     ./data/prediction.csv
#     model_best.chkpt
python anomaly_detection.py -data_pkl ./data/prediction.pkl -raw_data ./data/prediction.csv -model ./model/model_best.chkpt -output prediction/Stud

