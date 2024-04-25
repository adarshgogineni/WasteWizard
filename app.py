import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired
from flask import jsonify
from flask import flash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

login_manager = LoginManager()
login_manager.init_app(app)

# Assuming user data is stored like this
users = {'user1': {'password': 'pass1'}, 'user2': {'password': 'pass2'}}


#check file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    error = None
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('camera'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', form=form, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        users[form.username.data] = {'password': form.password.data}
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    file = request.files.get('file')  # This is a safer way to access the files
    if file:
        filepath = os.path.join('uploads', secure_filename(file.filename))
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully!'})
    return jsonify({'message': 'No file found'}), 400




@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
