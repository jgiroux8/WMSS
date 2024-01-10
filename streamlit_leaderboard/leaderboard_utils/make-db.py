import bcrypt 
from models import db


def make_password(password):
    bytes = password.encode('utf-8') 
    salt = bcrypt.gensalt() 
    hashed_password = bcrypt.hashpw(bytes, salt).decode('utf-8')
    return hashed_password

db.drop_all()

db.create_all()

from models import Team, User

# creating teams - leave commented until morning of 10/14
team_names = []
for t in range(1, 11):
    team_names.append(f"Team {t}")

# creating 4 passwords

passwords = ["AI4EIC"]*len(team_names)
# creating users,
# Note that there could be people with same name in different teams
users = dict()
for player in range(0, 41, 4):
    users[f"Team {player//4 + 1}"] = [f"Player_{player + 1}", f"Player_{player + 2}", f"Player_{player + 3}", f"Player_{player + 4}"]

db.create_all()
for team, pword in zip(team_names, passwords):
    pword = make_password(pword)
    print (team)
    db.session.add(Team(name = team, password = pword))
    db.session.commit()
    for user in users[team]:
        db.session.add(User(username = user, 
                            teamname = team, 
                            password = pword, 
                            OPENAI_API_KEY = "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")
                       )
        db.session.commit()

