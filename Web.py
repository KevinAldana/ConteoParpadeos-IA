# This is a sample Python script.
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
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
    while True:
        ret, frame =cap.read()

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
    app.run(debug=True)

