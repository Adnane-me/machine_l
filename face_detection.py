import dlib
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math


def face_detection(lien):
    def comparedistance(t, t1):
       d=0
       for i in range(len(t)):
            d = d + abs(t[i]-t1[i])
       return d
    def Knn(face_encoding, known_face_name, known_face_encoding):
     face_encoding_array = face_encoding.flatten()
     dictdistance = {}
     for name, encoding in zip(known_face_name, known_face_encoding):
        d = comparedistance(encoding, face_encoding_array)
        if name in dictdistance:
            dictdistance[name].append(d)
        else:
            dictdistance[name] = [d]
     print(dictdistance)
  
     sorted_dict = {k: sorted(v) for k, v in dictdistance.items()}
     print(sorted_dict)
  
     closest_face_distance = None
     closest_face_name = None
     for name, distances in sorted_dict.items():
        if closest_face_distance is None or distances[0] < closest_face_distance:
            closest_face_distance = distances[0]
            closest_face_name = name
    
     print("la distance", closest_face_distance - 4.50)
    
     if closest_face_distance - 4.50 <= 0.10:
        return closest_face_name
     else:
        return "Unknown"
    
    # maria detection 
    image_maria = face_recognition.load_image_file("photo/maria.jpg")
    maria_encoding_face = face_recognition.face_encodings(image_maria)[0]

    image_maria1 = face_recognition.load_image_file("photo/maria1.jpg")
    maria_encoding_face1 = face_recognition.face_encodings(image_maria1)[0]
    #shakira
    image_shakira= face_recognition.load_image_file("photo/shakira.jpg")
    shakira_encoding_face = face_recognition.face_encodings(image_shakira)[0]

    image_shakira1 = face_recognition.load_image_file("photo/shakira2.jpg")
    shakira_encoding_face1 = face_recognition.face_encodings(image_shakira1)[0]
    #neymar
    
    image_neymar = face_recognition.load_image_file("photo/neymar.jpg")
    image_neymar_face = face_recognition.face_encodings(image_neymar)[0]

    image_neymar1 = face_recognition.load_image_file("photo/neymar1.jpg")
    image_neymar1_encoding_face1 = face_recognition.face_encodings(image_neymar1)[0]
    #ronaldo
    
    image_ronaldo = face_recognition.load_image_file("photo/ronaldo3.jpg")
    image_ronaldo_face = face_recognition.face_encodings(image_ronaldo)[0]

    image_ronaldo1 = face_recognition.load_image_file("photo/ronaldo1.jpg")
    image_ronaldo1_encoding_face1 = face_recognition.face_encodings(image_ronaldo1)[0]
    # messi  

    image_messi = face_recognition.load_image_file("photo/messi.jpg")
    image_messi_face = face_recognition.face_encodings(image_messi)[0]

    image_messi1 = face_recognition.load_image_file("photo/messi1.jpg")

    image_messi1_encoding_face1 = face_recognition.face_encodings(image_messi1)[0]  

    # moutouali 

    image_mohssine1 = face_recognition.load_image_file("photo/mohssine1.jpg")
    image_mohssine1_face = face_recognition.face_encodings(image_mohssine1)[0]

    image_mohssine3 = face_recognition.load_image_file("photo/mohssine3.jpg")

    image_mohssine3_encoding_face1 = face_recognition.face_encodings(image_mohssine3)[0]  

    #boufal
    image_soufiane = face_recognition.load_image_file("photo/soufiane.jpg")
    image_soufiane_face = face_recognition.face_encodings(image_soufiane)[0]

    image_soufiane1 = face_recognition.load_image_file("photo/soufiane1.jpg")
    image_soufiane1_encoding_face1 = face_recognition.face_encodings(image_soufiane1)[0]  
    #youssef
    image_youssef = face_recognition.load_image_file("photo/youssef.jpg")
    image_youssef_face = face_recognition.face_encodings(image_youssef)[0]

    image_youssef2 = face_recognition.load_image_file("photo/youssef2.jpg")
    image_youssef2_encoding_face1 = face_recognition.face_encodings(image_youssef2)[0]  
    #saadLamjarred
    image_saadLamjarred = face_recognition.load_image_file("photo/saadLamjarred.jpg")
    image_saadLamjarred_face = face_recognition.face_encodings(image_saadLamjarred)[0]

    image_saadLamjarred2 = face_recognition.load_image_file("photo/saadLamjarred1.jpg")
    image_saadLamjarred2_encoding_face1 = face_recognition.face_encodings(image_saadLamjarred2)[0]
    #Mbappe
    image_Mbappe = face_recognition.load_image_file("photo/Mbappe.jpg")
    image_Mbappe_face = face_recognition.face_encodings(image_Mbappe)[0]

    image_Mbappe2 = face_recognition.load_image_file("photo/Mbappe1.jpg")
    image_Mbappe2_encoding_face1 = face_recognition.face_encodings(image_Mbappe2)[0]
    #badrHari
    image_badrHari = face_recognition.load_image_file("photo/badrHari.jpg")
    image_badrHari_face = face_recognition.face_encodings(image_badrHari)[0]

    image_badrHari2 = face_recognition.load_image_file("photo/badrHari1.jpg")
    image_badrHari2_encoding_face1 = face_recognition.face_encodings(image_badrHari2)[0]
    #hakimZiyech
    image_hakimZiyech = face_recognition.load_image_file("photo/ziyech1.jpg")
    image_hakimZiyech_face = face_recognition.face_encodings(image_hakimZiyech)[0]

    image_hakimZiyech2 = face_recognition.load_image_file("photo/ziyech2.jpg")
    image_hakimZiyech2_encoding_face1 = face_recognition.face_encodings(image_hakimZiyech2)[0]
    

    known_face_encoding = [maria_encoding_face ,maria_encoding_face1 ,shakira_encoding_face,shakira_encoding_face1,image_neymar_face,image_neymar1_encoding_face1,image_ronaldo_face,image_ronaldo1_encoding_face1,
                           image_messi_face ,image_messi1_encoding_face1, image_mohssine1_face, image_mohssine3_encoding_face1,image_soufiane_face,image_soufiane1_encoding_face1,image_youssef_face,image_youssef2_encoding_face1,
                           image_saadLamjarred_face, image_saadLamjarred2_encoding_face1,image_Mbappe_face,image_Mbappe2_encoding_face1,image_badrHari_face,image_badrHari2_encoding_face1,image_hakimZiyech_face,
                           image_hakimZiyech2_encoding_face1
                           ]
    known_face_name = ["maria", "maria","shakira","shakira" ,"neymar","neymar","ronaldo","ronaldo","messi", "messi", "mouhssine", "mouhssine","soufiane","soufiane","youssef","youssef","saadLamjarred","saadLamjarred",
                        "Mbappe","Mbappe","badrHari","badrHari","hakimZiyech","hakimZiyech"
                       ]

   # image_number = input("Please enter image number: ")
    #unknown_image = face_recognition.load_image_file(f"./uknouwn/{number}.jpg")
    unknown_image = face_recognition.load_image_file(lien)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    known_face_encoding_array = [i.flatten() for i in known_face_encoding]
    face_names = [Knn(face_encoding, known_face_name, known_face_encoding_array) for face_encoding in face_encodings]

    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        bold_font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", size=30)
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0))
        text_width, text_height = draw.textsize(name)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=bold_font)
    del draw

    return pil_image,name

