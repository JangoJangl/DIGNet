# DIGNet




## Inference TODO

## Training

### Download Data 

Download the Acronym dataset, ShapeNet meshes and make them watertight, following these [steps](https://github.com/NVlabs/acronym#using-the-full-acronym-dataset).

Download the training data consisting of 10000 table top training scenes with contact grasp information from [here](https://drive.google.com/drive/folders/1eeEXAISPaStZyjMX8BHR08cdQY4HF4s0?usp=sharing) and extract it to the same folder:

```
acronym
├── grasps
├── meshes
├── scene_contacts
└── splits
```

### Training DIGNet TODO

## Tools

### Generate Contact Grasps and Scenes yourself (optional) Contact-GraspNet Style

Thanks to [ NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet)

The `scene_contacts` downloaded above are generated from the Acronym dataset. To generate/visualize table-top scenes yourself, also pip install the [acronym_tools]((https://github.com/NVlabs/acronym)) package in your conda environment as described in the acronym repository.

In the first step, object-wise 6-DoF grasps are mapped to their contact points saved in `mesh_contacts`

```
python tools/create_contact_infos.py /path/to/acronym
```

From the generated `mesh_contacts` you can create table-top scenes which are saved in `scene_contacts` with

```
python tools/create_table_top_scenes.py /path/to/acronym
```

Takes ~3 days in a single thread. Run the command several times to process on multiple cores in parallel.

You can also visualize existing table-top scenes and grasps

```
python tools/create_table_top_scenes.py /path/to/acronym \
       --load_existing scene_contacts/000000.npz -vis
```

### acronym scripts

Follow the [Acronym Datset docs](https://github.com/NVlabs/acronym) and replace the paths

* --mesh_root ../dataset/acronym
* --object ../dataset/acronym/grasp

In general notice, that all `data` is moved to `../dataset/acronym`

Note: Just use a implemented model as orientation. You will do it! :)