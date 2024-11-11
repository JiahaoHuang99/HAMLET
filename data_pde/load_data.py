def load_data(DATASET_NAME, dataset_params):
    """
    select the dataset
    """

    # Darcy flow Dataset
    if DATASET_NAME in ['DARCY_FLOW_PDEBench_D3']:
        from data_pde.dataset_darcy_flow_pdebench_d3 import LoadDarcyFlowDatasetDGL as dataset_class

    # Diffusion Reaction Dataset
    elif DATASET_NAME in ['DIFFUSION_REACTION_2D_D3', 'DIFFUSION_REACTION_2D_PDEBench_D3']:
        from data_pde.dataset_diffusion_reaction_2d_pdebench_d3 import LoadDiffusionReaction2DDatasetDGL as dataset_class

    # Shallow Water Dataset
    elif DATASET_NAME in ['SHALLOW_WATER_2D_D3', 'SHALLOW_WATER_2D_PDEBench_D3']:
        from data_pde.dataset_shallow_water_2d_pdebench_d3 import LoadShallowWater2DDatasetDGL as dataset_class

    else:
        raise ValueError('Dataset {} not found'.format(DATASET_NAME))

    return dataset_class(dataset_params)