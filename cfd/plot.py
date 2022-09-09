import matplotlib.pyplot as plt
import numpy as onp

def plot_lcd_contour(vel,xc,yc,Re=100,interval=10):
    u = vel[:,:,0]
    v = vel[:,:,1]

    fig,ax = plt.subplots(1,1)
    ax.quiver(xc[::interval,::interval], yc[::interval,::interval],
              u[::interval,::interval], v[::interval,::interval],color='white',scale=5)
    pcm = ax.imshow(((u**2+v**2)**0.5).T,origin='lower',cmap='turbo',vmax=1.,vmin=0.,extent=[0.,1.,0.,1.])
    ax.set_title(f'Lid-driven-cavity: Re={Re}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm,ax=ax,label='velocity')
    fig.show()
    
    
def plot_lcd_compare(vel,xc,yc,Re=100,legend='calculated'):
    data_u = onp.genfromtxt('./data/u_Ghia_1982.csv',delimiter=',',skip_header=1)
    data_v = onp.genfromtxt('./data/v_Ghia_1982.csv',delimiter=',',skip_header=1)

    Re_num = onp.array([100,400,1000,3200,5000,7500,10000])
    ind = onp.where(Re_num==Re)[0]

    if ind.size == 0:
        print(f'error: Benchmark data for Re = {Re} does not exist!')
        return 0
    
    x_mid = vel.shape[0]
    y_mid = vel.shape[1]

    if x_mid % 2 == 0:
        u_mid = (vel[int(x_mid/2),:,0] + vel[int(x_mid/2)-1,:,0])/2
    else:
        u_mid = vel[int((x_mid-1)/2),:,0]

    if y_mid % 2 == 0:
        v_mid = (vel[:,int(y_mid/2),1] + vel[:,int(y_mid/2)-1,1])/2
    else:
        v_mid = vel[:,int((y_mid-1)/2),1]
        
    fig,axs = plt.subplots(2,1,figsize=(6,7))
    axs[0].plot(u_mid,yc[0,:,0])
    axs[0].plot(data_u[:,ind],data_u[:,0],'*')
    axs[0].set_xlabel('u at middle x')
    axs[0].set_ylabel('y')
    axs[0].legend([legend,'Ghia et al. - 1982'])

    axs[1].plot(v_mid,xc[:,0,0])
    axs[1].plot(data_v[:,ind],data_v[:,0],'*')
    axs[1].set_xlabel('v at middle y')
    axs[1].set_ylabel('x')
    axs[1].legend([legend,'Ghia et al. - 1982'])
    
    fig.show()