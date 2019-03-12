# SLAM
### tracking and reconstruction
- normally sperated into tracking pipeline and reconstruction pipeline.
- some work is to build reconstruction directly. [ElasticFusion](http://www.thomaswhelan.ie/Whelan16ijrr.pdf)
### tracking
- frame-to-frame
- frame-to-map
- map-to-map

### representation
- feature points (SURF, SIFT, ORB, etc)
- Fusion : surface (based on TSDFs)

# SFM 
not in real time 
- run global
- run one by one as timestramp


# 3d_segmentation

see the wiki page for more [WIKI](https://github.com/gggliuye/3d_segmentation/wiki)


3d slam demo is [here](https://github.com/gggliuye/3d_vision/blob/master/rgbd_camera_track/camera%20track%20rgbd.ipynb)

loop closure demo is [here](https://github.com/gggliuye/3d_vision/blob/master/Loop_closure_BOW.ipynb)

# connect with voxblox

need to change the coordinate system. (in rgbd_zed.yaml)


## PFE 

Titre et objectif du stage : Exploration des algorithmes de deep-learning pour la correspondance entre la vision et un espace 3D
lieur: Domont
Entreprise : Ideaversal est une société de conseil en innovation. Ses clients sont principalement des banques, sociétés immobilières et industriels du métal. Mission : L'objectif est d'explorer les algorithmes de deep-learning pour la correspondance entre la vision et un espace 3D sémantique. Il s'agit de réaliser un espace 3D contenant des objets labellisés (tels que tomates, courgettes, mauvaises herbes...) à partir d'une ou plusieurs caméras 2D ainsi que de capteurs (infra-rouge, ultra-son...). Le stagiaire devra définir des cas d'utilisation pour la mise en oeuvre d'un prototype en tenant compte de la facilité de réalisation et des co?ts. Puis il choisira une plateforme de mise en oeuvre de deep-learning (theano, caffe, tensorflow...) adaptée à ce projet. Il devra ensuite installer la ou les plateformes choisies sur des machines avec GPU et éventuellement réaliser des comparaisons entre ces algorithmes. Puis il développera un modèle de deep learning et réalisera des tests sur un cas concret.
