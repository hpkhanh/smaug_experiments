sweep_params = {
    # Number of accelerators. Any integer value.
    "num_accels": [1, 2],
    # SoC interface choice. Only DMA or ACP.
    "soc_interface": ["dma", "acp"],
    # L2 cache size. Any power of 2 value like 64K, 128K..
    "l2_size": [65536],
    # L2 associativity. Any power of 2 value.
    "l2_assoc": [4],
    # Accelerator clock time (ns). Integers from 1 to 10.
    "acc_clock": [1],
    # Main memory type. Supported types:
    # ["LPDDR3_1600_1x32", "LPDDR2_S4_1066_1x32", "HBM_1000_4H_1x128",
    #  "HBM_1000_4H_1x64", "GDDR5_4000_2x32", "DDR4_2400_8x8", "DDR4_2400_4x16",
    #  "DDR4_2400_16x4", "DDR3_1600_8x8", "DDR3_2133_8x8"]
    "mem_type": ["LPDDR4_3200_2x16"],
    # CPU clock time (ns). Float in [0.25, 0.3, 0.4, 0.5, 0.8, 1], representing
    # 4GHz, 3.3GHz, 2.5GHz, 2GHz, 1.25GHz and 1GHz respectively.
    "cpu_clock": [0.5],
    # Use pipelined DMA or not. 0 or 1.
    "pipelined_dma": [1],
    # Ignore cache flush on DMA. 0 or 1.
    "ignore_cache_flush": [0],
    # Invalidate cache on a DMA store. 0 or 1.
    "invalidate_on_dma_store": [1],
    # Maximum number of outstanding DMA requests. Integer in [16, 32, 64, 128].
    "max_dma_requests": [16],
    # Number of DMA channels in the DMA controller. Integer in [1, 2, 4, 8].
    "num_dma_channels": [1],
    # DMA chunk size. Integer in [32, 64, 128].
    "dma_chunk_size": [64],
}
