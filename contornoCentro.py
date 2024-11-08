import cv2 as cv
import numpy as np

# Carregar a imagem
im = cv.imread('image.png')
assert im is not None, "Arquivo não pode ser lido, verifique o caminho."
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Converter para escala de cinza

# Inverter a imagem para destacar áreas não brancas
im_invert = cv.bitwise_not(imgray)

# Aplicar threshold para criar uma imagem binária (considerando não brancos)
ret, thresh = cv.threshold(im_invert, 127, 255, cv.THRESH_BINARY)

# Encontrar contornos na imagem
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Verificar se há contornos e calcular o centro do primeiro contorno
if contours:
    cnt = contours[0]  # Pegar o primeiro contorno
    
    # Calcular os momentos do contorno
    M = cv.moments(cnt)
    
    # Calcular o centro
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Marcar o centro na imagem
        cv.circle(im, (center_x, center_y), 5, (0, 0, 0), -1)
        print(f"Centro do contorno: ({center_x}, {center_y})")
    else:
        print("Contorno tem área zero.")

    # Classificar formas
    for c in contours:
        # Aproximar contornos
        epsilon = 0.01 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        
        # Obter as coordenadas do retângulo delimitador
        x, y, w, h = cv.boundingRect(approx)

        # Identificar a forma
        if len(approx) == 3:
            cv.putText(im, 'Triangulo', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            print('Triangulo')
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                cv.putText(im, 'Quadrado', (x , y - 5), 1, 1.5, (0, 255, 0), 2)
                print('Quadrado')
            else:
                cv.putText(im, 'Retangulo', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
                print('Retangulo')
        elif len(approx) == 5:
            cv.putText(im, 'Pentagono', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            print('Pentagono')
        elif len(approx) == 6:
            cv.putText(im, 'Hexagono', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            print('Hexagono')
        elif len(approx) > 10:
            cv.putText(im, 'Circulo', (x, y - 5), 1, 1.5, (0, 255, 0), 2)
            print('Circulo')
        else:
            print("Nao foi encontrado nenhuma forma geometrica")
            

    
# Desenhar todos os contornos em verde
cv.drawContours(im, contours, -1, (0, 255, 0), 3)

# Mostrar a imagem final
cv.imshow("Centro do Contorno e forma geometerica ", im)
cv.waitKey(0)
cv.destroyAllWindows()
