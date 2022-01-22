import face_recognition
import cv2
import numpy as np
import pickle

f = open("ref_name.pkl", "rb")
ref_dictt = pickle.load(f)
f.close()
f = open("ref_embed.pkl", "rb")
embed_dictt = pickle.load(f)
f.close()

known_face_encodings = []  # encodingd of faces
known_face_names = []  # ref_id of faces

for ref_id, embed_list in embed_dictt.items():
    for embed in embed_list:
        known_face_encodings += [embed]
        known_face_names += [ref_id]

# FACE DETECTION AND RECOGNITION PART
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# GENDER RECOGNITION VARIABLES
genderProto = "deploy_gender.prototxt"
genderModel = "gender_net.caffemodel"
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
gender = ''


#This value is constant.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20


def highlightFace(net, frame, conf_threshold=0.9):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]


            face_names.append(name)


    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        pass

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        

   
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (93, 222, 163), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        if name in ref_dictt:
            cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            font = cv2.FONT_HERSHEY_DUPLEX
        else:
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, gender, (left + 50, bottom - 300), font, 1.0, (255, 255, 255), 1)
        font = cv2.FONT_HERSHEY_DUPLEX

    cv2.imshow('Video', frame)
    cv2.waitKey(1)




