jax.config.update("jax_enable_x64", True)

def pure_cfd():
    start_time = time.time()
    for i in range(0, 100):
        example.time_integration()
        elapsed = time.time() - start_time
        print(f'time:{elapsed},T_max:{example.T0.max()},vmax:{np.linalg.norm(example.vel0,axis=3).max()}')
        start_time = time.time()