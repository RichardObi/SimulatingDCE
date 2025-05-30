import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        # Iterate over the B_paths and check if the corresponding A_path exists
        # If not, remove the B_path from the list
        if opt.isTrain:
            self.B_paths_new = [B_path for B_path in self.B_paths if os.path.join(self.dir_A, os.path.basename(B_path).replace('0005', '0000').replace('0004', '0000').replace('0003', '0000').replace('0002', '0000').replace('0001', '0000').replace('CONCAT', '0000')) in self.A_paths]
            print(f'Warning: Removed {len(self.B_paths) - len(self.B_paths_new)} B_paths that do not have a corresponding A_path')
            self.B_paths = self.B_paths_new

            # TODO: Only CONCAT replaced here. Needs to be adjusted in case of DCE phases 1-5
            # Test if string concat is in any of the A_paths
            if 'CONCAT' in self.B_paths[20] and 'CONCAT' in self.B_paths[30]:
                self.A_paths_new = [A_path for A_path in self.A_paths if os.path.join(self.dir_B, os.path.basename(A_path).replace('0000', 'CONCAT')) in self.B_paths]
                print(f'Warning: Removed {len(self.A_paths) - len(self.A_paths_new)} A_paths that do not have a corresponding B_path')
                self.A_paths = self.A_paths_new

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            # check if B_paths is defined for index (sometimes the same input maps to multiple outputs some of which are not defined - e.g. DCE phase 3 and 4)
            if index >= len(self.B_paths):
                print(f'Warning: No B_path for index {index} in B_paths of length {len(self.B_paths)}')
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'