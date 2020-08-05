sweep_params = {
    # Number of accelerators. Any integer value.
    "num_accels": [1, 2],
    # SoC interface choice. Only DMA or ACP.
    "soc_interface": ["dma", "acp"],
    # L2 cache size. Any power of 2 value like 64K, 128K..
    "l2_size": [65536],
    # L2 associativity. Any power of 2 value.
    "l2_assoc": [4],
    # Accelerator clock time. Integers from 1 to 10.
    "acc_clock": [1],
    # Main memory type. Supported types:
    # ["LPDDR3_1600_1x32", "LPDDR2_S4_1066_1x32", "HBM_1000_4H_1x128",
    #  "HBM_1000_4H_1x64", "GDDR5_4000_2x32", "DDR4_2400_8x8", "DDR4_2400_4x16",
    #  "DDR4_2400_16x4", "DDR3_1600_8x8", "DDR3_2133_8x8"]
    "mem_type": ["LPDDR4_3200_2x16"],
}
