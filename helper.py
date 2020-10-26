from flask import request, jsonify
import jwt
from werkzeug.security import generate_password_hash, check_password_hash 
from functools import wraps
from classe import Users
import datetime
import random
import string

def user_by_username(username):
    try:
        return Users.query.filter(Users.email == username).one()
    except:
        return None

def auth():
    auth = request.get_json()
    if not auth or not auth['username'] or not auth['password']:
        return jsonify({'message':"Informações faltando",'WWW-Authenticate':'Basic auth= "Login Required"'}), 401
    
    user = user_by_username(auth['username'])

    if not user:
        return jsonify({'message':"Usuário não encontrado",'data':{} } ), 401
    if user and check_password_hash(user.password, auth['password']):
        token = jwt.encode({'username': user.email, 'exp':datetime.datetime.now() + datetime.timedelta(hours=12) }, 'wNOUCuB0gePM')
            
        return jsonify({'message':'Login Realizado com Sucesso', 'token':token.decode('utf-8'), 'exp': datetime.datetime.now() + datetime.timedelta(hours=12) })

def token_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        token = request.headers['Authorization'].split(' ')[1]
        print(token)
        if not token:
            return jsonify({'message':'Token is invalid', 'data': {} }), 401
        try:
            data = jwt.decode(token,'wNOUCuB0gePM')
            current_user = user_by_username(username=data['username'])
        except:
            return jsonify({'message':'Token está expirado', 'data': {} }), 401
        
        return f(current_user,*args,**kwargs)
    return decorated