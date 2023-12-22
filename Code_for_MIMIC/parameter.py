#import RAFDB_AU, AffectNet, FERPlus


AU_img_folder_path="/home/FaceData/RAF-AU/aligned/"
AU_train_list_file = "/home/FaceData/RAF-AU/aligned_split_train_label.txt"
AU_test_list_file = "/home/FaceData/RAF-AU/aligned_split_test_label.txt"
RAFAU_FER_label = '/home/FaceData/RAF-AU/RAFAU_FER-labels.csv'

FER_TestlabelPath = "/home/FaceData/RAF-DB/label/test.txt"
FER_TrainlabelPath = "/home/FaceData/RAF-DB/label/train.txt"
FER_img_folder_path ="/home/FaceData/RAF-DB/aligned/"
RAFDB_AU_CSV =  "/home/FaceData/RAF-DB/RAFDB_AU_label_OpenFace.csv"

EmotioNetAU_img_folder_path="/home/FaceData/EmotioNet/224_shrink/"
# EmotioNetAU_img_folder_path="/home/ssd7T/FaceData/EmotioNet/align224/"
EmotioNetAU_train_list_file="/home/FaceData/EmotioNet/label/label_align_224_train.txt"
EmotioNetAU_test_list_file ="/home/FaceData/EmotioNet/label/label_align_224_test.txt"
EmotioNet_FER_label = '/home/FaceData/EmotioNet/EmotioNet_FER-labels.csv'

FERPlus__img_folder_path="/home/FaceData/FERPlus/FERPlus_shrink/"
# FERPlus__img_folder_path="/home/ssd7T/FaceData/FERPlus/data-aligned/"
FERPlus_AU_CSV="/home/FaceData/FERPlus/FERPlus_AU_label_OpenFace.csv"
FERPlus_AU_CSV_valid = "/home/FaceData/FERPlus/FERPlus_AU_label_OpenFace_valid.csv"
FERPlus_train_img="/home/FaceData/FERPlus/data-aligned/FER2013Train/"
FERPlus_valid_img="/home/FaceData/FERPlus/data-aligned/FER2013Valid/"

AffectNet_img_path="/home/zf/dataset/FaceData/AffectNet/align/"
AffectNet_train_folder_7="/home/zf/dataset/FaceData/AffectNet/split_7"
AffectNet_train_folder="/home/zf/dataset/FaceData/AffectNet/split"
AffectNet_train_npy="/home/zf/dataset/FaceData/AffectNet/training.npy"
AffectNet_validation_npy="/home/zf/dataset/FaceData/AffectNet/validation.npy"
AffectNet_7_FER_AU_CSV = "/home/zf/dataset/FaceData/AffectNet/AffectNet_FER_AU_label.csv"
AffectNet_7_FER_AU_CSV_valid = "/home/zf/dataset/FaceData/AffectNet/AffectNet_FER_AU_label_valid.csv"