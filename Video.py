from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import time
# Creamos nuestra funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

#Realizar la video captura
cap = cv2.VideoCapture(0)

#Funcion generacion frames
def gen_frame():
    #Variables
    parpadeo = False
    conteo = 0
    tiempo = 0
    inicio = 0
    final = 0
    conteo_sue = 0
    muestra = 0
    while True:
        ret, frame = cap.read()

        #Listas
        px = []
        py = []
        lista = []

        if not ret:
            break
        else:
            # Correccion de color
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Observamos los resultados
            resultados = MallaFacial.process(frameRGB)
            # Si tenemos rostros
            if resultados.multi_face_landmarks:
                # Iteramos
                for rostros in resultados.multi_face_landmarks:
                    # Dibujamos
                    mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

                    for id, puntos in  enumerate(rostros.landmark):
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)

                        lista.append([id, x, y])
                        if len(lista) == 468:
                            #Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]

                            longitud1= math.hypot(x2-x1, y2-y1)

                           #Ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]

                            longitud2= math.hypot(x4-x3, y4-y3)

                            cv2.putText(frame, f'Parpadeos: {int(conteo)}',(30, 60), cv2.FONT_HERSHEY_SIMPLEX,1,
                                        (0,255,0),2)
                            cv2.putText(frame, f'Microsueños: {int(conteo_sue)}', (380, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            cv2.putText(frame, f'Duracion: {int(muestra)}', (210, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)

                        # Ifs
                            if longitud1<=10  and longitud2 <=10 and parpadeo == False:
                                conteo = conteo+1
                                parpadeo = True
                                inicio = time.time()
                            elif longitud2 >10 and longitud1 >10  and parpadeo == True:
                                parpadeo = False
                                final = time.time()
                                tiempo =  round(final-inicio, 0 )

                            if tiempo >=3:
                                conteo_sue = conteo_sue+1
                                muestra = tiempo
                                inicio = 0
                                final = 0

            #Codificamos  nuestro video  en bytes
            suc, encode = cv2.imencode('.jpg',frame)
            frame = encode.tobytes()
        yield (b'--frame\r\n'b'content-Type: image/jpeg\r\n\r\n'+frame+ b'\r\n')
#Creamos la app
app = Flask(__name__)

#RutaPrincipal
@app.route('/')
def index():
    return render_template('Index.html')
@app.route('/video')
def video():
    return Response(gen_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')
#Ejecutar el server
if __name__ == "__main__":
    app.run( debug=True)