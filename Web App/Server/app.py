from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
import smtplib
from  flask_mail import Mail,Message
import dotenv
import re
from flask import session, redirect, flash
from flask import Flask, request, render_template, url_for, send_from_directory
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from email.mime.text import MIMEText
import random





load_model = tf.keras.models.load_model

dotenv.load_dotenv()
app = Flask(__name__, template_folder="template", static_folder="static")

# connect to mysql database 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mri_prosses'
mysql = MySQL(app)  
app.secret_key = 'a4d2fb3a20c94731b6dcfae8b1e3d0b2'

@app.route('/sign_up', methods=["GET", "POST"])
def sign_up():
    if request.method == "POST":
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            
            cursor = mysql.connection.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                         (name, email, password))
            mysql.connection.commit()
            return redirect(url_for('login'))
            
        except Exception as e:
            mysql.connection.rollback()
            flash("Registration failed")
            return redirect(url_for('sign_up'))

    return render_template('Sign_up.html')




@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE (username = %s OR email = %s) AND password = %s", (username,username,password))

        user = cursor.fetchone()

        if user:
            user_email = user[2]  # ⚠️ Adjust index based on your DB schema
            otp_code = send_otp_to_email(user_email)

            # Store OTP and username in session for later verification
            session['otp'] = otp_code
            session['username'] = username

            flash("An OTP has been sent to your email. Please verify.")
            return redirect(url_for("two_factor"))
        else:
            flash("Invalid username or password")
            return render_template("Login.html")

    return render_template("Login.html")

#EMAIL sending message
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('DEL_EMAIL')
app.config['MAIL_PASSWORD'] = os.getenv('PASSWORD')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)



def send_otp_to_email(email):
    otp = str(random.randint(100000, 999999))  # generate 6-digit code

    msg = Message("Your Login OTP Code", sender=os.getenv("DEL_EMAIL"), recipients=[email])
    msg.body = f"Your OTP code is: {otp}"

    mail.send(msg)
    return otp


@app.route("/contact", methods=["POST"])
def contact():
    if request.method == "POST":
        try:
            name = request.form["name"]
            email = request.form["email"]  # <-- استقبال البريد الإلكتروني
            subject = request.form["subject"]
            message = request.form["message"]
            
            # تكوين الرسالة - يمكن تضمين البريد الإلكتروني في نص الرسالة
            msg = Message(subject, sender=os.getenv("DEL_EMAIL"), recipients=[os.getenv("REC_EMAIL")])
            msg.body = f"Hello from {name} ({email}),\n\n{message}"
            
            mail.send(msg)
            flash("Your message has been sent successfully.")
        except Exception as e:
            flash(f"An error occurred while sending the email: {str(e)}")

        return redirect(url_for("home"))


# Load models
segmentation_model = load_model(r"C:\Users\MSI GAMER\Desktop\Third Year\Pfe\Code\Saves\Seg6\best_model6.h5", compile=False)
classification_model = load_model(r"C:\Users\MSI GAMER\Desktop\Third Year\Pfe\Code\Saves\Cls_ga\best_model.h5", compile=False)

class_labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SEG_FOLDER = os.path.join(BASE_DIR, 'static', 'segmented')
os.makedirs(UPLOAD_FOLDER, exist_ok=True, mode=0o777)
os.makedirs(SEG_FOLDER, exist_ok=True, mode=0o777)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEG_FOLDER'] = SEG_FOLDER

def preprocess_for_classification(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 128, 128, 1)
    return img_array

@app.route("/home", methods=["GET", "POST"])
def home():
    
    
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filename = file.filename
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(upload_path)

                # --- Segmentation ---
                img = Image.open(upload_path).convert("L")
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=-1)
                input_tensor = tf.expand_dims(img_array, axis=0)

                pred_mask = segmentation_model.predict(input_tensor)[0]
                mask = (pred_mask[:, :, 0] * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask).resize(img.size)

                seg_filename = f"seg_{filename}"
                seg_path = os.path.join(app.config['SEG_FOLDER'], seg_filename)
                mask_img.save(seg_path)

                # --- Classification ---
                class_input = preprocess_for_classification(upload_path)
                class_prediction = classification_model.predict(class_input)
                class_index = np.argmax(class_prediction)
                predicted_class = class_labels[class_index]

                return render_template("index.html",
                    original=url_for('static', filename=f"uploads/{filename}"),
                    segmented=url_for('static', filename=f"segmented/{seg_filename}"),
                    tumor_type=predicted_class,
                )

            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                return f"An error occurred: {str(e)}", 500
        else:
            return "Invalid file format. Please upload a valid image (PNG, JPG, or JPEG).", 400

    return render_template("index.html")


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def index():
    return redirect(url_for('login')) 

@app.route('/2FA', methods=["GET"])
def two_factor():
    return render_template('2FA.html')

@app.route('/verify_otp', methods=["POST"])
def verify_otp():
    # Combine all the parts of the OTP
    user_otp = ''.join([
        request.form.get('otp1'),
        request.form.get('otp2'),
        request.form.get('otp3'),
        request.form.get('otp4'),
        request.form.get('otp5'),
        request.form.get('otp6')
    ])

    if 'otp' in session and user_otp == session['otp']:
        flash("OTP verified successfully!")
        return redirect(url_for("home"))  # or any dashboard
    else:
        flash("Invalid OTP. Please try again.")
        return redirect(url_for("two_factor"))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)