from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from tools import Reactor
from rdkit.Chem import Draw
from rdkit import Chem
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from xenonpy.descriptor import Fingerprints
import json

fp_type = ['RDKitFP', 'AtomPairFP', 'MACCS',
           'ECFP', 'FCFP', 'TopologicalTorsionFP']


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens), smi
    return ' '.join(tokens)


def get_time():
    now = datetime.utcnow()
    name = ''.join([str(a) for a in [now.year, now.month, now.day,
                                     now.hour, now.minute, now.second, now.microsecond]])
    return name


app = Flask(__name__, static_url_path='/static', static_folder='static')
print(app.root_path)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

db = SQLAlchemy(app)


class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    R1 = db.Column(db.String(300), nullable=False)
    R2 = db.Column(db.String(300), nullable=False)
    P1 = db.Column(db.String(300), nullable=False)
    P2 = db.Column(db.String(300), nullable=False)
    P3 = db.Column(db.String(300), nullable=False)
    P4 = db.Column(db.String(300), nullable=False)
    P5 = db.Column(db.String(300), nullable=False)
    P1DC = db.Column(db.Float(200), nullable=False, default=0)
    P1GTT = db.Column(db.Float(200), nullable=False, default=0)
    P2DC = db.Column(db.Float(200), nullable=False, default=0)
    P2GTT = db.Column(db.Float(200), nullable=False, default=0)
    P3DC = db.Column(db.Float(200), nullable=False, default=0)
    P3GTT = db.Column(db.Float(200), nullable=False, default=0)
    P4DC = db.Column(db.Float(200), nullable=False, default=0)
    P4GTT = db.Column(db.Float(200), nullable=False, default=0)
    P5DC = db.Column(db.Float(200), nullable=False, default=0)
    P5GTT = db.Column(db.Float(200), nullable=False, default=0)
    note = db.Column(db.String(500), nullable=False, default='Nothing left')
    analysis = db.Column(db.Integer, nullable=False, default=0)
    utctime = db.Column(db.String(300), nullable=False, default=get_time())

    def __repr__(self):
        return 'Blog post ' + str(self.id)


def reaction(R1, R2):
    ChemicalReactor = Reactor()
    ChemicalReactor.BuildReactor(model_list=[
                                 './models/transformer_models/STEREO_mixed_augm_model_average_20.pt'], max_length=100, n_best=5)

    reactant = [smi_tokenizer(R1) + " . " + smi_tokenizer(R2)]
    _, products = ChemicalReactor.react(reactant)

    products = [line.replace(" ", "") for line in products[0]]
    # product = products[0]
    return (products)


def property_plot(smile, plot_path):
    P_fp = Fingerprints(featurizers=fp_type, input_type='smiles',
                        on_errors='nan').transform([smile])
    P_fp = P_fp.dropna()

    PolyGeno_property = pd.read_csv(
        "./data/forward_train_Polymer_genome_new_asterisk_R.csv")
    DC = np.ravel(PolyGeno_property['Dielectric.Constant'])
    GTT = np.ravel(PolyGeno_property['Glass.Transition.Temperature'])
    singleFP2dc = pickle.load(
        open('./models/P_SMILES2DC_ElasticNet.sav', 'rb'))
    singleFP2gtt = pickle.load(
        open("./models/P_SMILES2GTT_ElasticNet.sav", 'rb'))
    property_region = {'Dielectric_Constant': [
        3, 4], 'Glass_Transition_Temperature': [300, 350]}
    property_minmax = {'Dielectric_Constant': [min(DC), max(
        DC)], 'Glass_Transition_Temperature': [min(GTT), max(GTT)]}

    plt.figure(figsize=(10, 10))
    plt.xlim(property_minmax["Dielectric_Constant"][0]-1,
             property_minmax["Dielectric_Constant"][1]+1)
    plt.ylim(property_minmax["Glass_Transition_Temperature"][0]-20,
             property_minmax["Glass_Transition_Temperature"][1]+20)
    plt.plot(DC, GTT, 'k.', markersize=5, alpha=0.4, label='existing data')
    if len(P_fp) == 1:
        dc_pre = singleFP2dc.predict(P_fp)
        gtt_pre = singleFP2gtt.predict(P_fp)
        plt.scatter(dc_pre, gtt_pre, s=250, c='g',
                    edgecolor='k', label='product')

    plt.xlabel("Dielectric Constant", fontsize=20)
    plt.ylabel("Glass Transition Temperature ($^\circ$C)", fontsize=20)
    plt.title("Properties of chemical structures", fontsize=20)
    plt.legend(loc='lower right', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot([property_region["Dielectric_Constant"][0],
              property_region["Dielectric_Constant"][0],
              property_region["Dielectric_Constant"][1],
              property_region["Dielectric_Constant"][1],
              property_region["Dielectric_Constant"][0]],
             [property_region["Glass_Transition_Temperature"][0],
              property_region["Glass_Transition_Temperature"][1],
              property_region["Glass_Transition_Temperature"][1],
              property_region["Glass_Transition_Temperature"][0],
              property_region["Glass_Transition_Temperature"][0]], 'r--')
    plt.savefig(plot_path)
    plt.close()
    if len(P_fp) == 0:
        dc_pre = 0
        gtt_pre = 0
    dc_pre = round(dc_pre[0], 2)
    gtt_pre = round(gtt_pre[0], 2)
    return (dc_pre, gtt_pre)


def smile2structure(smile):
    mol = [Chem.MolFromSmiles(smile)]
    im = Draw.MolsToGridImage(
        [mol[0]], molsPerRow=1, subImgSize=(200, 200))
    return im


def draw_result(post):
    print(post.id)
    print(post.utctime)
    R1_im = smile2structure(post.R1)
    R1_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'R1.png')
    R2_im = smile2structure(post.R2)
    R2_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'R2.png')
    P1_im = smile2structure(post.P1)
    P1_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'P1.png')
    P2_im = smile2structure(post.P2)
    P2_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'P2.png')
    P3_im = smile2structure(post.P3)
    P3_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'P3.png')
    P4_im = smile2structure(post.P4)
    P4_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'P4.png')
    P5_im = smile2structure(post.P5)
    P5_im.save('/Users/qi/Desktop/flask_app/static/images/' +
               str(post.id)+'/'+post.utctime+'P5.png')
    P1DC, P1GTT = property_plot(
        smile=post.P1, plot_path='/Users/qi/Desktop/flask_app/static/images/' + str(post.id)+'/'+post.utctime+'P1y.png')
    P2DC, P2GTT = property_plot(
        smile=post.P2, plot_path='/Users/qi/Desktop/flask_app/static/images/' + str(post.id)+'/'+post.utctime+'P2y.png')
    P3DC, P3GTT = property_plot(
        smile=post.P3, plot_path='/Users/qi/Desktop/flask_app/static/images/' + str(post.id)+'/'+post.utctime+'P3y.png')
    P4DC, P4GTT = property_plot(
        smile=post.P4, plot_path='/Users/qi/Desktop/flask_app/static/images/' + str(post.id)+'/'+post.utctime+'P4y.png')
    P5DC, P5GTT = property_plot(
        smile=post.P5, plot_path='/Users/qi/Desktop/flask_app/static/images/' + str(post.id)+'/'+post.utctime+'P5y.png')
    return (P1DC, P2DC, P3DC, P4DC, P5DC, P1GTT, P2GTT, P3GTT, P4GTT, P5GTT)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/posts')
def posts():
    # if request.method == 'POST':
    #     post_R1 = request.form['R1']
    #     post_R2 = request.form['R2']
    #     post_P = reaction(post_R1, post_R2)
    #     post_note = request.form['note']
    #     new_post = BlogPost(
    #         R1=post_R1, R2=post_R2, Product1=post_P, note=post_note)
    #     db.session.add(new_post)
    #     db.session.commit()
    #     return redirect('/posts')
    # else:
    all_posts = BlogPost.query.order_by(BlogPost.id).all()
    return render_template('posts.html', posts=all_posts)


@app.route('/posts/delete/<int:id>')
def delete(id):
    post = BlogPost.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    return redirect('/posts')


@app.route('/posts/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    post = BlogPost.query.get_or_404(id)
    if request.method == 'POST':
        post.R1 = request.form['R1']
        post.R2 = request.form['R2']
        post.note = request.form['note']
        products = reaction(post.R1, post.R2)
        post.P1 = products[0]
        post.P2 = products[1]
        post.P3 = products[2]
        post.P4 = products[3]
        post.P5 = products[4]
        post.analysis = 0
        post.utctime = get_time()
        db.session.commit()
        return redirect('/posts')
    else:
        return render_template('edit.html', post=post)


@app.route('/posts/info/<int:id>', methods=['GET', 'POST'])
def info(id):
    post = BlogPost.query.get_or_404(id)
    if request.method == 'POST':
        print('yes')
        post.analysis = 1
        P1_DC, P2_DC, P3_DC, P4_DC, P5_DC, P1_GTT, P2_GTT, P3_GTT, P4_GTT, P5_GTT = draw_result(
            post)
        post.P1DC = P1_DC
        post.P2DC = P2_DC
        post.P3DC = P3_DC
        post.P4DC = P4_DC
        post.P5DC = P5_DC
        post.P1GTT = P1_GTT
        post.P2GTT = P2_GTT
        post.P3GTT = P3_GTT
        post.P4GTT = P4_GTT
        post.P5GTT = P5_GTT
        db.session.commit()

        return render_template('info.html', post=post)
    else:
        return render_template('info.html', post=post)


@app.route('/posts/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        post_R1 = request.form['R1']
        post_R2 = request.form['R2']
        post_note = request.form['note']
        products = reaction(post_R1, post_R2)
        post_P1 = products[0]
        post_P2 = products[1]
        post_P3 = products[2]
        post_P4 = products[3]
        post_P5 = products[4]
        new_post = BlogPost(
            R1=post_R1, R2=post_R2, P1=post_P1, P2=post_P2, P3=post_P3, P4=post_P4, P5=post_P5, note=post_note)
        db.session.add(new_post)
        db.session.commit()
        Path("/Users/qi/Desktop/flask_app/static/images/"+str(new_post.id)
             ).mkdir(parents=True, exist_ok=True)
        # print("alrady submit")
        return redirect('/posts')
    else:
        return render_template('new_post.html')


if __name__ == "__main__":
    app.run(debug=True)
