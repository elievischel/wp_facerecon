import io
from os import listdir, remove
from os.path import isfile, join, splitext
import json
import face_recognition
import numpy
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import ast

# Global storage for images
faces_dict = {}
persistent_faces = "faces/"
wordpress_instance = "http://localhost/web1/"

# Create flask app
app = Flask(__name__)
CORS(app)


# <Picture functions> #

def is_picture(filename):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions


def get_all_picture_files(path):
    files_in_dir = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return [f for f in files_in_dir if is_picture(f)]


def remove_file_ext(filename):
    return splitext(filename.rsplit('/', 1)[-1])[0]


def calc_face_encoding(image):
    # Currently only use first face found on picture
    loaded_image = face_recognition.load_image_file(image)
    faces = face_recognition.face_encodings(loaded_image)

    # If more than one face on the given image was found -> error
    if len(faces) > 1:
        raise Exception(
            "Found more than one face in the given training image.")

    # If none face on the given image was found -> error
    if not faces:
        raise Exception("Could not find any face in the given training image.")

    return faces[0]

def detect_faces_in_image(request):
    faces_dict = get_all_face_encodings()

    file = extract_image(request)
    # Load the uploaded image file
    img = face_recognition.load_image_file(file)

    # Get face encodings for any faces in the uploaded image
    uploaded_faces = face_recognition.face_encodings(img)

    # Defaults for the result object
    faces_found = len(uploaded_faces)
    faces = []

    if faces_found:
        face_encodings = list(faces_dict.values())
        for uploaded_face in uploaded_faces:
            match_results = face_recognition.compare_faces(
                face_encodings, uploaded_face)
            match = False

            arraytostring = numpy.array2string(uploaded_face, separator=',')

            for idx, match in enumerate(match_results):
                if match:
                    match = list(faces_dict.keys())[idx]
                    match_encoding = face_encodings[idx]
                    dist = face_recognition.face_distance([match_encoding], uploaded_face)[0]
                    faces.append({
                        "id": match,
                        "dist": dist,
                        "face_encoding": arraytostring
                    })
                    faces_dict.clear()
                    faces_dict = get_all_face_encodings()
            if not match:
                if faces_found == 1:
                    try:
                        faces.append({
                            "id": file.filename,
                            "face_encoding": arraytostring,
                        })
                        faces_dict.clear()
                        faces_dict = get_all_face_encodings()

                    except Exception as exception:
                        raise BadRequest(exception)

    return {
        "count": faces_found,
        "faces": faces
    }


def extract_image(request):
    # Check if a valid image file was uploaded
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file


# <Picture functions> #

# <API functions> #


# <API functions> #

# get all users faces from wp
def get_all_face_encodings():
    url = "https://planyourmove.ch/wp-json/wp/v2/posts"

    headers = {
        'Authorization': 'Basic YWRtaW46YWRtaW4='
    }
    response = requests.request("GET", url, headers=headers)
    json_object = response.json()
    return dict([(face[0], numpy.array(ast.literal_eval(face[1]))) for face in json_object])

# <Controller>

@app.route('/', methods=['POST'])
def web_recognize():
    file = extract_image(request)

    if file and is_picture(file.filename):
        # The image file seems valid! Detect faces and return the result.
        return jsonify(detect_faces_in_image(request))
    else:
        raise BadRequest("Given file is invalid!")


@app.route('/faces', methods=['GET', 'POST', 'DELETE'])
def web_faces():
    # GET
    if request.method == 'GET':
        return jsonify(list(faces_dict.keys()))

    # POST/DELETE
    file = extract_image(request)
    if 'id' not in request.args:
        raise BadRequest("Identifier for the face was not given!")

    if request.method == 'POST':
        jsonify(detect_faces_in_image(file))
        # TODO validate if user exists already

        app.logger.info('%s loaded', file.filename)
        # HINT jpg included just for the image check -> this is faster then passing boolean var through few methods
        # TODO add method for extension persistence - do not forget about the deletion
        file.save("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))
        try:
            new_encoding = calc_face_encoding(file)
            faces_dict.update({request.args.get('id'): new_encoding})
        except Exception as exception:
            raise BadRequest(exception)

    elif request.method == 'DELETE':
        faces_dict.pop(request.args.get('id'))
        remove("{0}/{1}.jpg".format(persistent_faces, request.args.get('id')))

    return jsonify(list(faces_dict.keys()))


@app.route('/', methods=['GET'])
def all_posts():
    get_all_face_encodings()
    return "welcome to the first facial recon for wordpress"


# </Controller>


if __name__ == "__main__":
    print("Starting by generating encodings for found images...")
    # Calculate known faces
    faces_dict = get_all_face_encodings()
    #print(faces_dict)

    # Start app
    print("Starting WebServer...")
    app.run(host='localhost', port=8080, debug=False)
