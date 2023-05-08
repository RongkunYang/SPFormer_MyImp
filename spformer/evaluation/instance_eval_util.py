import json


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id==-1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return instance_id//1000

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances==instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict['instance_id'] = self.instance_id
        dict['instance_id'] = self.instance_id
        dict['label_id'] = self.label_id
        dict['vert_count'] = self.vert_count
        dict['med_dist'] = self.med_dist
        dict['dist_conf'] = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id = int(data['instance_id'])
        self.label_id = int(data['label_id'])
        self.vert_count = int(data['vert_count'])
        if ('med_dist' in data):
            self.med_dist = float(data['med_dist'])
            self.dist_conf = float(data['dist_conf'])

    def __str__(self):
        return '(' + str(self.instance_id) + ')'