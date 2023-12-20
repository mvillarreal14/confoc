# Imports
from fastai.conv_learner import *
from pathlib import Path
from scipy import ndimage

class FeatureSaver():
    ''' Forward Hook Technique: We need to create a class in which the following 
        statment is executed:
        
            self.hook = model.register_forward_hook(self.hook_fn)
            
        Where:
            - self.hook: is an attribute
            - self.hook_fn: is a method that takes as paramenters a model, its input 
              and the correspondng output.
              
        When the method self.hook_fn is called it saves the output in an accessor 
        attributed of the class. 
        
        The class also includes another methond called self.hook.remove() to stop the 
        hooking process.
    '''
    features=None
    
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output): 
        self.features = output
        
    def close(self): 
        self.hook.remove()

class StyledImageGenerator():
    ''' Given an image x and and Image Style s, this class allows the creation of the following:
            - x_c: Content of x
            - x_s: Styled version of x using x_c and stle s: x_s = x_c + s.
    ''' 
    
    def __init__(self, x, s, model=None, tfm=None, layers=None, layer_idx=None, max_iter=None, 
                 show_iter=None, opt_alg=None, content_loss_scale=1e6, gram_scale=1e6, verbose=True):
        ''' This function is the Consturctor.
            Args:
                x: Original Image X.
                s: Style Image S.
                model: Model used to do the generation.
                tfm: Transformation to be applied to the image. Generally it chages its size.
                layers: Layers to obtain the output activations for Content and Style.
                layer_idx: Index of the layer to obtain the output activation for Content.
                max_iter: Maximum number of iterations during the optimization.
                show_iter: Specifies what iteration and its values should be displayed.
                opt_alg: Optimization algorightm.
            Returns:
                None
        '''
        # Check required fields
        if model == None or tfm == None:
            raise Exception('Model and transformation are required.')
        if layers == None or layer_idx == None: 
            raise Exception('Layers for extraction of style and id of layer for content are rquired.')
        if max_iter == None or show_iter == None:
            raise Exception('Number of iterations and frequency of displaying are required.')
        # Initialize variables
        self.x = None if x is None else open_image(x)
        self.s = None if s is None else open_image(s)
        self.model = model
        self.tfm = tfm
        self.layers = layers
        self.layer_idx = layer_idx
        self.max_iter = max_iter
        self.show_iter = show_iter
        self.opt_alg = 'lgbfs' if opt_alg is None or type(opt_alg) != type("str") else opt_alg
        self.optimizer = None
        self.n_iter = None
        self.tensor_to_optimize = None
        self.fs = None
        self.content_target = None if x is None else self.get_content_target()
        self.style_target = None if s is None else self.get_style_target()
        self.content_loss_scale = content_loss_scale 
        self.gram_scale = gram_scale
        self.verbose = verbose
        
    def set_x(self, x):
        ''' This function sets a new value for X and gets the target features for content.
            Args: 
                s: New value of S
            Returns:
                None.
        '''
        if x is not None:
            self.x = open_image(x)
            self.content_target = self.get_content_target()
        else:
            self.x = None 
            self.content_target = None
        
    def set_s(self, s):
        ''' This function sets a new value for S and gets the target features for style.
            Args: 
                s: New value of S
            Returns:
                None.
        '''
        if s is not None:
            self.s = open_image(s)
            self.style_target = self.get_style_target()
        else:
            self.s = None 
            self.style_target = None
            
    def set_content_loss_scale(self, loss_scale):
        ''' This function sets a new value for the content loss scale
            Args: 
                loss_scale: New loss scale. Type: Integer.
            Returns:
                None.
        '''
        self.content_loss_scale = loss_scale
        
    def set_loss_scale_automatically(self, option=None):
        ''' This function sets a new value for the content loss scale
            Args: 
                loss_scale: New loss scale. Type: Integer.
            Returns:
                None.
        '''
        if self.x is not None:
            h = self.x.shape[0]
            w = self.x.shape[1]
            loss_scale = 0
            
            # Multiplication
            if option is None or option == 'area':
                r = h*w
                if (r < 50**2): loss_scale = 1e7
                if (r >= 50**2  and r < 80**2) : loss_scale = 1e6
                if (r >= 80**2  and r < 110**2): loss_scale = 5e5    
                if (r >= 110**2 and r < 170**2): loss_scale = 3e5
                if (r >= 170**2 and r < 200**2): loss_scale = 3e5
                if (r >= 200**2): loss_scale = 1e5
             
            # Average length
            if option == "mean_lengtn":
                r = (h+w)/2
                if (r < 50): loss_scale = 1e7
                if (r >= 50  and r < 80) : loss_scale = 1e6
                if (r >= 80  and r < 110): loss_scale = 5e5    
                if (r >= 110 and r < 170): loss_scale = 3e5
                if (r >= 170 and r < 200): loss_scale = 3e5
                if (r >= 200): loss_scale = 1e5
            
            # Minimum length 
            if option == "min_length":
                if (h < 50) or (w < 50): loss_scale = 1e7
                if (h >= 50 and h < 80)   or (w >= 50 and w < 80)  : loss_scale = 1e6
                if (h >= 80  and h < 110) or (w >= 80  and w < 110): loss_scale = 5e5    
                if (h >= 110 and h < 180) or (w >= 110 and w < 170): loss_scale = 3e5
                if (h >= 170 and h < 200) or (w >= 170 and w < 200): loss_scale = 3e5
                if (h >= 200) or (h >= 200): loss_scale = 1e5
            
            self.content_loss_scale = loss_scale 
            
        else:
            print('No changes doned: input x is None.')
            
    def set_gram_scale(self, gram_scale):
        ''' This function sets a new value for the gram scale
            Args: 
                gram_scale: New gram scale. Type: Integer.
            Returns:
                None.
        '''
        self.gram_scale = gram_scale
        
    def set_layer_idx(self, layer_idx):
        ''' This function sets the layer id from which to extract feature for content generation
            Args: 
                layer_idx: Layer id. Type: Integer.
            Returns:
                None.
        '''
        self.layer_idx = layer_idx
          
    def get_content_target(self):
        ''' This function gets the content target
            Args: 
                None
            Returns:
                features: Output actications of the spefified layer (only one layer).
        '''
        fs = self.get_feature_saver(self.layer_idx)
        self.model(VV(self.tfm(self.x)[None]))
        features = fs.features
        fs.close()
        return features
        
    def get_style_target(self):
        ''' This function gets the style target (output of the different layers given S as input)
            Args: 
                None
            Returns:
                features: Output actications of the spefified layers (multiple layers).
        '''
        fs = self.get_feature_saver()
        img_tfm = self.tfm(self.x) if self.x is not None else self.tfm(self.s) # new
        self.model(VV(self.tfm(self.match_image_size(np.rollaxis(img_tfm,0,3), self.s))[None]))           # <---- HERE
        #self.model(VV(self.tfm(self.s)[None]))
        features = []
        for o in fs:
            features.append(o.features)
            o.close()
        return features
        
    def get_feature_saver(self, idx=None):
        ''' This function produces an instance of Feture Saver
            Args: 
                idx: Index of the layer from which to get the output features.
            Returns:
                fs: Instance of FeatureSaver. 
        '''
        if idx == None:
            fs = [FeatureSaver(children(self.model)[layer]) for layer in self.layers]
        else:
            fs =  FeatureSaver(children(self.model)[self.layers[idx]])
        return fs
        
    def get_tensor_to_optimize(self):
        ''' This function produces a random image from a uniform distribution.
            Args:
                is_brighter: Defines whether to do the division by 2 before or after the 
                transformation Type: Boolean.
                img: Image that specifies the size of the random image being 
                creted. Type: Image.
                tfms: Transformation applied to the new random image. Type: Transformation 
            Returs:
            img_random: Random Image. Type: Image.
        '''   
        img_tfm = self.tfm(self.x) if self.x is not None else self.tfm(self.s)                            # <---- HERE
        img_random = np.random.uniform(0, 1, size=np.rollaxis(img_tfm,0,3).shape).astype(np.float32)
        img_random = scipy.ndimage.filters.median_filter(img_random, [8,8,1])
        img_random = self.tfm(img_random)
        return V(img_random[None], requires_grad=True)
    
    def match_image_size(self, src, targ):
        ''' This function resizes targ to the size of src.
            Args:
                src: Image to be resized. Type: Image.
                targ: Reference image to the the size. Type: Image.
            Returs:
                res: resized image. Type: Image.
        '''  
        sh,sw,_ = src.shape
        th,tw,_ = targ.shape
        rat = max(sh/th,sw/tw);
        res = cv2.resize(targ, (int(tw*rat), int(th*rat)))
        return res[:sh,:sw]
        
    def pick_optimizer(self):
        ''' This function picks the optimizer. At this moment between LGBFS and Adam. Other 
            optimizers wll be added later.
            Args: 
                None
            Returns:
                optimizer: Eiether LGBFS or Adam
        '''
        if self.opt_alg == 'lgbfs':
            optimizer = optim.LBFGS([self.tensor_to_optimize], lr=0.5)
        else:
            optimizer = optim.Adam([self.tensor_to_optimize], lr=0.1, 
                                        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer
 
    def closure(self, loss_fn):
        ''' This function is the closure for the optimizer LGBFS
            Args: 
                loss_fn: Loss function used to to the optimization
            Returns:
                loss: Loss after a cycle of the LGBFS agrithm
        '''
        self.optimizer.zero_grad()
        loss = loss_fn(self.tensor_to_optimize)
        loss.backward()
        self.n_iter += 1
        if self.n_iter%self.show_iter==0 and self.verbose: 
            print(f'Iteration: {self.n_iter}, loss: {loss.item()}')
        #if self.n_iter%self.show_iter==0: print(f'Iteration: {self.n_iter}, loss: {loss}')
        return loss
    
    def update(self, loss_fn):
        ''' This function does the update with the specified optimizer
            Args: 
                loss_fn: Loss function used to to the optimization
            Returns:
                None
        '''
        self.n_iter = 0
        while self.n_iter <= self.max_iter:
            if self.opt_alg == 'lgbfs':
                self.optimizer.step(partial(self.closure, loss_fn))
            else:
                self.optimizer.zero_grad()
                loss = loss_fn(self.tensor_to_optimize)
                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                if self.n_iter%self.show_iter==0 and self.verbose: 
                    print(f'Iteration: {self.n_iter}, loss: {loss.item()}')
                #if self.n_iter%self.show_iter==0: print(f'Iteration: {self.n_iter}, loss: {loss}')
                    
    def gram(self, tensor):
        ''' Gram Matrix: Matrix of the dot product of the flattened channels of a layers's
            activations. This is a non-spatial representation of how the channels correlate 
            with each other. The diagonal of this matrix tells what channels are the most 
            active. The off-diagonal says which channels appear together. Overall, if two 
            pictures have the same style, we expect that in some layer activations they will 
            have similar gram matrices. The diagonal alone might be used to get the style,
            but here we use the entire matrix
            Args: 
                tensor: Tensor with several channels
            Returns:
                gram: Gram matrix of the provided tensor
        '''
        b,c,h,w = tensor.size()
        x = tensor.view(b, c, -1)
        gram = torch.bmm(x, x.transpose(1,2))/(c*h*w)*self.gram_scale 
        return gram
    
    def gram_mse_loss(self, x_hat, x):
        ''' This function calculate the MSE loss between the gram matrices of two tensors. 
            Args: 
                x: Tensor to be optimized.
            Returns:
                comb_loss: Combined loss as explained above.
        '''
        return F.mse_loss(self.gram(x_hat), self.gram(x))
    
    def comb_loss_fn(self, x):                                                               
        ''' This function combines the content_loss_fn() and style_loss_fn() to estimate the loss 
            of the styled image.
            Args: 
                x: Tensor to be optimized.
                loss_scale: Multiplier of loss to adjust quality. Type: Integer.
            Returns:
                comb_loss: Combined loss as explained above.
        '''
        self.model(x)
        outs = [o.features for o in self.fs]
        content_loss = F.mse_loss(outs[self.layer_idx], self.content_target)*self.content_loss_scale       
        style_loss = sum([self.gram_mse_loss(o, t) for o, t in zip(outs, self.style_target)])
        comb_loss = content_loss + style_loss
        return comb_loss

    def style_loss_fn(self, x):
        ''' This function calculates the sum of the MSE losses in each layer between the gram 
            matrix produced by the tensor_to_optimize and the gram matrix produced by the 
            style S.
            Args: 
                x: Tensor to be optimized.
            Returns:
                style_loss: Gram-MSE loss as explained above.
        '''
        self.model(x)
        outs = [o.features for o in self.fs]
        losses = [self.gram_mse_loss(o, t) for o, t in zip(outs, self.style_target)]
        style_loss = sum(losses)
        return style_loss
    
    def content_loss_fn(self, x):                                           
        ''' This function calculates the MSE loss between the features  produced by the 
            tensor_to_optimize and the features produced by the input X.
            Args: 
                x: Tensor to be optimized.
                loss_scale: Multiplier of loss to adjust quality. Type: Integer.
            Returns:
                content_loss: MSE loss as explained above.
        '''
        self.model(x)
        content_loss = F.mse_loss(self.fs[self.layer_idx].features, self.content_target)*self.content_loss_scale  
        return content_loss
    
    def optimize_random_image(self, loss_fn):
        ''' This function generates a random image and optimize it using the received loss function
            Args: 
                loss_fn: Loss fuction to be used for the optimization
            Returns:
                tensor_to_optimize: Optimized tensor
        '''
        self.fs = self.get_feature_saver()
        self.tensor_to_optimize = self.get_tensor_to_optimize()
        self.optimizer = self.pick_optimizer()
        self.update(loss_fn)
        for o in self.fs: o.close()
        self.fs = None
        return self.tensor_to_optimize
    
    def get_content(self):
        ''' This function gets the content of X. It calls the function optimize_random_image() passing the
            proper loss function.
            Args: 
                None
            Returns: 
                content: Content of image X
        '''
        content = self.optimize_random_image(self.content_loss_fn)
        content = self.tfm.denorm(np.rollaxis(to_np(content.data),1,4))[0]
        return content
        
    def get_style(self):
        ''' This function gets the style from S. It calls the function optimize_random_image() passing the
            proper loss function.
            Args: 
                None
            Returns: 
                style: Style from image S
        '''
        style = self.optimize_random_image(self.style_loss_fn)
        style = self.tfm.denorm(np.rollaxis(to_np(style.data),1,4))[0]
        return style
        
    def get_styled_image(self):
        '''This function gets the styled image Xs from X and S. It calls the function optimize_random_image() 
            passing the proper loss function
            Args: 
                None
            Returns: 
                styled: Styled Image from imageX and style S
        '''
        styled = self.optimize_random_image(self.comb_loss_fn)
        styled = self.tfm.denorm(np.rollaxis(to_np(styled.data),1,4))[0]
        return styled