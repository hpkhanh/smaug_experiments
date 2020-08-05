sweep_params = {
    #"num_accels": [1, 2, 4, 8],
    #"soc_interface": ["dma", "acp"],
    #"l2_size": [
    #    65536, 131072, 256 * 1024, 512 * 1024, 1024 * 1024, 2048 * 1024,
    #    4096 * 1024
    #],
    #"l2_assoc": [4, 8, 16, 32],
    #"acc_clock": [1, 2, 4, 8],
    "num_accels": [1],
    "soc_interface": ["dma"],
    "l2_size": [2048 * 1024],
    "l2_assoc": [16],
    "acc_clock": [1],
    "mem_type": [
        "LPDDR4_3200_2x16", "LPDDR3_1600_1x32", "LPDDR2_S4_1066_1x32",
        "HBM_1000_4H_1x128", "HBM_1000_4H_1x64", "GDDR5_4000_2x32",
        "DDR4_2400_8x8", "DDR4_2400_4x16", "DDR4_2400_16x4", "DDR3_1600_8x8",
        "DDR3_2133_8x8"
    ]
}
