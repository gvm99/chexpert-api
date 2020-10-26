from database import db

class Historico(db.Model):
    __tablename__ = 'exames'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    cpf = db.Column(db.String())
    tipoExame = db.Column(db.String())
    response = db.Column(db.String())

    def __init__(self, name, cpf, tipoExame, response):
        self.name = name
        self.cpf = cpf
        self.tipoExame = tipoExame
        self.response = response

    def __repr__(self):
        return '<id {}>'.format(self.id)
    
    def serialize(self):
        return {
            'id': self.id, 
            'name': self.name,
            'cpf': self.cpf,
            'tipoExame':self.tipoExame,
            'response' : self.response
        }

class Users(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    cpf = db.Column(db.String())
    crm = db.Column(db.String())
    email = db.Column(db.String())
    password = db.Column(db.String())

    def __init__(self, name, cpf, crm,email, password):
        self.name = name
        self.cpf = cpf
        self.crm = crm
        self.email = email
        self.password = password

    def __repr__(self):
        return '<id {}>'.format(self.id)
    
    def serialize(self):
        return {
            'id': self.id, 
            'name': self.name,
            'cpf': self.cpf,
            'crm':self.crm,
            'email' : self.email,
            'password' : self.password
        }