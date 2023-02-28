# Object Representation Generation




## Idea

This package extends the acronym dataset with object embeddings of given objects. 


## Embedding Generation

By running the [create_trainings_dataset.py](create_trainings_dataset.py) you create a folder with .npz-archieves with 
all embeddings of included models. You can adjust the model list in the [config.yaml](config.yaml). You have to adjust
the path of your acronym dataset there as well.
Structure will be as follows:
```
acronym
├── grasps
├── meshes
├── scene_contacts
├── splits
├── embedddings <- genrated by script
```
The format of the npz-files is a dict with the Encoder-name as key and the Embeddings as data.
The script ads for each object a new reference in the corresponding .hfd5-file in [acronym](acronym).


### Extend Models 

You have to create a new folder and wit the following structure.
```
Encoder-Name
├── cfgs
├── checkpoints
├── wrapper.py
├── misc # when needed
```
In cfg are all configuration files saved your model needs and transformations you need to preprocess.
In checkpoints are the checkpoint.pth saved for your encoder.
The wrapper.py implements a Class which deals with your model. The parent class is implemented in inference.py
It is important to implement all passed methods. Furthermore the Name of the Class in wrapper.py has to match the name 
in [config.yaml](config.yaml).
In misc you can put all mandatory files for you need for instanciating the Encode-model.

