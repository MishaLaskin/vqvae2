
imsize48_default_architecture = dict(
    conv_args=dict(
        kernel_sizes=[5, 3, 3],
        n_channels=[16, 32, 64],
        strides=[3, 2, 2],
        res_h_dim=32,
        n_res_layers=3,
    ),
    conv_kwargs=dict(
        hidden_sizes=[],
        batch_norm_conv=False,

    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3, 3],
        n_channels=[32, 16],
        strides=[2, 2],
        res_h_dim=32,
        n_res_layers=3,
    ),
    deconv_kwargs=dict(
        batch_norm_deconv=False,
        batch_norm_fc=False,
    ),
    pixelcnn_kwargs=dict(
        n_layers=15,
    )
)
