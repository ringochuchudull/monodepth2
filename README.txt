This is a modification of original repo 'monodepth2' so that it can be used as an API wrapper
The original authors perserves the right to monodepth2

Requirement:
    PyTorch

install:
    python setup.py install

Usages:
    Full usages can also be found in 'example_monodepth_usage.ipynb'
    
    ``` Example Python script
    from monodepth2.usemonodepth import load_mondepthWeight, get_depthDispar, get_depthdispar_simplified, only_get_disparity
    
    encoder, depth_decoder, monoDepth_config, targetDevice = load_mondepthWeight(model_name='mono+stereo_640x192')
    
    cv2img = cv2.imread(SampleImg) 
    dis, scale_disp, colimg1, pillowim = get_depthdispar_simplified(input_image=cv2img,
                                                                    model=(encoder, depth_decoder),
                                                                    modelConfig=monoDepth_config,
                                                                    device=targetDevice, 
                                                                    channel_last=True,
                                                                    isPil=False)
    ```

Last date of modeification: 24/1/2020