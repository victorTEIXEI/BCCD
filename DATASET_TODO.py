import numpy as np
import cv2
import os
import glob

pasta_alt='C:/Users/teixe/OneDrive/Imagens/saved pictures alterada/'
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
pasta='C:/Users/teixe/OneDrive/Imagens/Saved Pictures'
files = glob.glob( pasta+ "/*.jpg", recursive=True)
print(files)

for i, file in enumerate(files):
    img=cv2.imread(pasta)
    # cv2.imshow("original",img)
    cv2.waitKey(0)

    #                    divisao dos tres canais
    canal_B,canal_G,canal_R = cv2.split(img)

    #                   clahe em cada canal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    clahe_B = clahe.apply(canal_B)
    clahe_G = clahe.apply(canal_G)
    clahe_R = clahe.apply(canal_R)

    juntando=cv2.merge((clahe_B,clahe_G,clahe_R))
    # cv2.imwrite("clahe_canais.jpg",juntando)

    bilateral = cv2.bilateralFilter(juntando, d=35, sigmaColor=75, sigmaSpace=75)
    # cv2.imwrite("bilateral.jpg",bilateral)
    # cv2.imshow("clahe em cada canal + bilateral",bilateral)
    cv2.waitKey(0)



    #               TENTANDO SUBTRAI CANAS R G

    canal_R, canal_G,canal_B = cv2.split(bilateral)

    #               subtraçao com cv2
    subtracao = cv2.subtract(canal_R,canal_G)
    mediana = cv2.medianBlur(subtracao,3)

    # cv2.imshow("subtracaoRG + mediana",mediana)
    cv2.waitKey(0)


    _, limiarizada_otsu = cv2.threshold(mediana, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Imagem Limiarizada", limiarizada_otsu)
    cv2.waitKey(0)


    fechamento = cv2.morphologyEx(limiarizada_otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # cv2.imshow("Fechamento", fechamento)
    cv2.waitKey(0)


    # inicio =(113,41)
    # fim = (265,200)
    # cv2.rectangle(img,inicio, fim, (0, 255, 0), 2)
    # cv2.imshow("retangulo na celula", img)
    # cv2.waitKey(0)





    contornos, _ = cv2.findContours(fechamento, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maior_contorno = max(contornos, key=cv2.contourArea)

    # Criação de uma caixa delimitadora com margem de 10 pixels
    x, y, w, h = cv2.boundingRect(maior_contorno)
    print(x,y,w,h)
    inicio = (x - 10, y - 10)
    fim = (x + w + 10, y + h + 10)
    cv2.rectangle(fechamento, inicio, fim, (255, 255, 255), 4)

    # cv2.imshow("Caixa Delimitadora na Célula", fechamento)
    cv2.waitKey(0)


    quadrado = img[y-10:y+h+10, x-10:x+w+10]
    # cv2.imshow('Quadrado Extraído', quadrado)
    cv2.waitKey(0)
        


    cv2.imwrite(pasta_alt + str(i) + ".jpg",quadrado)
