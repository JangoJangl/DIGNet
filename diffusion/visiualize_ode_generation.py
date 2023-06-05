from math import fabs
import os
import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from networks.k_diffusion.classifier_free_sampler import ClassifierFreeSampleModel
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import einops 
from matplotlib.ticker import NullFormatter, MaxNLocator
from numpy import linspace
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)

# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos,ypos=x0,y0
    radm,radn=ra,rb
    an=ang
    co,si=np.cos(an),np.sin(an)
    the=linspace(0,2*np.pi,Nb)
    X=radm*np.cos(the)*co-si*radn*np.sin(the)+xpos
    Y=radm*np.cos(the)*si+co*radn*np.sin(the)+ypos
    return X,Y


def main(model_store_path: str) -> None:
    
    use_max_values = False
    data_range = 2 
    make_gif = True
    n_inference_steps = 3
    gif_duration = 1
    random_idx = 172
    noise_scheduler = 'exponential'
    safe_individual_plot = True
    window_zoom = 1
    zoom_loc = 'lower left'
    filenames_2 = []
    filenames = []
    
    # make new subdir
    file_path = os.path.join(model_store_path, f'{n_inference_steps}_steps_idx_{random_idx}')
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    # get the path to the config yaml file
    cfg_store_path = os.path.join(model_store_path,'.hydra/config.yaml')
    # get path to the bet repo -> required for the block push environment if needed
    os.environ['ASSET_PATH'] = "/home/moritz/code/bet"
    # init wandb logger and config from hydra path 
    cfg = OmegaConf.load(cfg_store_path) 
    # load the required classes to train and test the agent 
    workspace_manager = hydra.utils.instantiate(cfg.workspaces)
    agent = hydra.utils.instantiate(cfg.agents)
    # get the scaler instance and set teh bounds for the sampler if required
    agent.get_scaler(workspace_manager.scaler)
    agent.set_bounds(workspace_manager.scaler)
    agent.load_pretrained_model(model_store_path)

    agent.sigma_max = 33
    agent.sigma_min = 0.39
    # get a random state and goal 
    state, action, mask, goal = workspace_manager.datasets[0][random_idx]
    # goal = workspace_manager.datasets[0][random_idx][3]
    state = np.random.uniform([9.5, 9.5], [10.5, 10.5], (1,2))
    goal =  np.random.uniform([9.5, 9.5], [10.5, 10.5], (1,2))
    goal = np.array([10, 15]).reshape(1,2)
    goal = np.array([15, 10]).reshape(1,2)
    goal = np.array([15, 15]).reshape(1,2)
    state = torch.from_numpy(state)
    goal = torch.from_numpy(goal)
    # state = einops.rearrange(torch.from_numpy(state), 'b d -> b 1 d')
    # goal = einops.rearrange(torch.from_numpy(goal), 'b d -> b 1 d')
    action = agent.scaler.scale_output(torch.from_numpy(action).reshape(1,2))
    agent.model.eval()
    # agent.model = ClassifierFreeSampleModel(agent.model, cond_lambda=0)
    # results = agent.visualize_ode(
    sigmas = agent.get_noise_schedule(5, 'exponential')
    sigmas = torch.flip(sigmas, [0])
    sigmas[0] = 0.01
    latent_action = agent.forward_ode(state, action, goal, sigmas)
    results = agent.predict(state,goal,
        # get_mean=500000, 
        # new_sampling_steps=n_inference_steps,
        # noise_scheduler=noise_scheduler,
    )
    '''# test forward stuff 
    forward_results = agent.visualize_ode(
        state,
        goal,
        get_mean=500000, 
        new_sampling_steps=n_inference_steps,
        noise_scheduler=noise_scheduler,
        start_x=results[-1][:500000,:],
        forward_ode=True,
        return_last_step=True
    )
    results = agent.visualize_ode(
        state,
        goal,
        get_mean=500000, 
        new_sampling_steps=n_inference_steps,
        noise_scheduler=noise_scheduler,
        start_x=forward_results,
        forward_ode=False,
        return_last_step=False
    )'''
    # scaled_action = agent.scaler.scale_output(action)
    # plotting stuff
    action = torch.tensor(((0.1, 1), (1, 0.1), (0.82, 0.66)))
    for i in range(n_inference_steps+1):
        
        # Set up your x and y labels
        xlabel = '$\mathrm{a_1}$' + f'({i})'
        ylabel = '$\mathrm{a_2}$'+ f'({i})'

        x = results[i][:, 0].cpu().numpy()
        y = results[i][:, 1].cpu().numpy()
        
        # Set up default x and y limits
        xlims = [min(x),max(x)]
        ylims = [min(y),max(y)]
        # Set up the size of the figure
        # Define the locations for the axes
        left, width = 0.12, 0.55
        bottom, height = 0.12, 0.55
        
        bottom_h = left_h = left+width+0.02
        fig = plt.figure(1, figsize=(8.5,9))
        
        rect_temperature = [left, bottom, width, height] # dimensions of temp plot
        rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
        rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
        # Make the three plots
        axTemperature = plt.axes(rect_temperature) # temperature plot
        axHistx = plt.axes(rect_histx) # x histogram
        axHisty = plt.axes(rect_histy) # y histogram
        
        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        
        # Find the min/max of the data
        # Find the min/max of the data
        xmin = min(xlims)
        xmax = max(xlims)
        ymin = min(ylims)
        ymax = max(ylims)
        
        # Make the 'main' temperature plot
        # Define the number of bins
        nxbins = 50
        nybins = 50
        nbins = 100
        
        xbins = linspace(start = xmin, stop = xmax, num = nxbins)
        ybins = linspace(start = ymin, stop = ymax, num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        
        # aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
        
        H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
        X = xcenter
        Y = ycenter
        Z = H
        
        # Plot the temperature data
        cax = (axTemperature.imshow(H, extent=[
                min(min(xlims), - x.std()*3 + x.mean()-1), 
                max(max(xlims),  x.std()*3 + x.mean()+1),
                min(min(ylims), - y.std()*3 + y.mean()-1), 
                max(max(ylims), y.std()*3 + y.mean()+1)
            ],
            interpolation='nearest', origin='lower',aspect=1))
        
        # Plot the temperature plot contours
        contourcolor = 'white'
        xcenter = np.mean(x)
        ycenter = np.mean(y)
        ra = np.std(x)
        rb = np.std(y)
        ang = 0
        
        X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
        axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points', horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25)
        
        X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
        axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points',horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25, color = contourcolor)
        
        X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
        axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points',horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25, color = contourcolor)
        
        #Plot the axes labels
        axTemperature.set_xlabel(xlabel,fontsize=25)
        axTemperature.set_ylabel(ylabel,fontsize=25)
        
        #Make the tickmarks pretty
        ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(18)
            label.set_family('serif')
        
        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(18)
            label.set_family('serif')
        
        #Set up the plot limits
        axTemperature.set_xlim(min(min(xlims), - x.std()*3 + x.mean()-1), 
                         max(max(xlims),  x.std()*3 + x.mean()+1) )
        axTemperature.set_ylim(min(min(ylims), - y.std()*3 + y.mean()-1), 
                         max(max(ylims), y.std()*3 + y.mean()+1))
        
        #Set up the histogram bins
        xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
        ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
        
        #Plot the histograms
        axHistx.hist(x, bins=xbins, color = 'blue')
        axHisty.hist(y, bins=ybins, orientation='horizontal', color = 'red')
        
        #Set up the histogram limits

        axHistx.set_xlim(min(min(xlims), - x.std()*3 + x.mean()-1), 
                         max(max(xlims),  x.std()*3 + x.mean()+1))
        axHisty.set_ylim(min(min(ylims), - y.std()*3 + y.mean()-1), 
                         max(max(ylims), y.std()*3 + y.mean()+1))
        
        #Make the tickmarks pretty
        ticklabels = axHistx.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(12)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = axHisty.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(12)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHistx.yaxis.set_major_locator(MaxNLocator(4))

        # plt.hist2d(x, y, bins=(100, 100))
        file_name = os.path.join(file_path, f'histogram_{i}_{n_inference_steps}_steps_idx_{random_idx}.png')
        if safe_individual_plot:
            plt.savefig(file_name)
        filenames_2.append(file_name)
        plt.close()
    
    # variant with same x y lim for all plots
    # Set up default x and y limits
    if use_max_values:
        xlims = [min(results[0][:, 0].cpu().numpy()),max(results[0][:, 0].cpu().numpy())]
        ylims = [min(results[0][:, 1].cpu().numpy()),max(results[0][:, 1].cpu().numpy())]
    
        xmin = min(xlims)
        xmax = max(xlims)
        ymin = min(ylims)
        ymax = max(ylims)
    else:
        xlims = [-data_range,data_range]
        ylims = [-data_range, data_range]
        xmin = -data_range
        xmax = data_range
        ymin = -data_range
        ymax = data_range
        
    for i in range(n_inference_steps+1):
        
        # Set up your x and y labels
        xlabel = '$\mathrm{a_1}$' + f'({i})'
        ylabel = '$\mathrm{a_2}$'+ f'({i})'

        x = results[i][:, 0].cpu().numpy()
        y = results[i][:, 1].cpu().numpy()

        # Set up the size of the figure
        # Define the locations for the axes
        left, width = 0.12, 0.55
        bottom, height = 0.12, 0.55
        
        bottom_h = left_h = left+width+0.02
        fig = plt.figure(1, figsize=(8.5,9))
        
        rect_temperature = [left, bottom, width, height] # dimensions of temp plot
        rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
        rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
        # Make the three plots
        axTemperature = plt.axes(rect_temperature) # temperature plot
        axHistx = plt.axes(rect_histx) # x histogram
        axHisty = plt.axes(rect_histy) # y histogram
        
        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        # Make the 'main' temperature plot
        # Define the number of bins
        nxbins = 50
        nybins = 50
        nbins = 100
        
        xbins = linspace(start = xmin, stop = xmax, num = nxbins)
        ybins = linspace(start = ymin, stop = ymax, num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        
        # aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
        
        H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
        X = xcenter
        Y = ycenter
        Z = H
        
        # Plot the temperature data
        cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
        interpolation='nearest', origin='lower',aspect=1))
        
        # Plot the temperature plot contours
        contourcolor = 'white'
        xcenter = np.mean(x)
        ycenter = np.mean(y)
        ra = np.std(x)
        rb = np.std(y)
        ang = 0
        
        X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
        axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points', horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25)
        
        X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
        axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points',horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25, color = contourcolor)
        
        X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
        axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
        axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                            textcoords='offset points',horizontalalignment='right',
                            verticalalignment='bottom',fontsize=25, color = contourcolor)
        
        #Plot the axes labels
        axTemperature.set_xlabel(xlabel,fontsize=25)
        axTemperature.set_ylabel(ylabel,fontsize=25)
        
        #Make the tickmarks pretty
        ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(18)
            label.set_family('serif')
        
        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(18)
            label.set_family('serif')
        
        #Set up the plot limits
        axTemperature.set_xlim(xlims)
        axTemperature.set_ylim(ylims)
        
        #Set up the histogram bins
        xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
        ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
        
        #Plot the histograms
        axHistx.hist(x, bins=xbins, color = 'blue')
        axHisty.hist(y, bins=ybins, orientation='horizontal', color = 'red')
        
        #Set up the histogram limits

        #Set up the histogram limits
        axHistx.set_xlim( xmin, xmax)
        axHisty.set_ylim( ymin, ymax)
        
        #Make the tickmarks pretty
        ticklabels = axHistx.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(12)
            label.set_family('serif')
        
        #Make the tickmarks pretty
        ticklabels = axHisty.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(12)
            label.set_family('serif')
        
        #Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHistx.yaxis.set_major_locator(MaxNLocator(4))

        # plt.hist2d(x, y, bins=(100, 100))
        file_name = os.path.join(file_path, f'histogram_{i}_v2_{n_inference_steps}_steps_idx_{random_idx}.png')
        if safe_individual_plot:
            plt.savefig(file_name)
        filenames.append(file_name)
        plt.close()
    
    fig, ax = plt.subplots(nrows=1, ncols=n_inference_steps+1, figsize=((n_inference_steps +1)*4, 4), dpi=300) 
    for i in range(n_inference_steps+1):
        if use_max_values:
            # xlims = [min(results[i][:, 0].cpu().numpy()),max(results[i][:, 0].cpu().numpy())]
            # ylims = [min(results[i][:, 1].cpu().numpy()),max(results[i][:, 1].cpu().numpy())]
            xmin = np.mean(results[i][:, 0].cpu().numpy()) - 2.5*np.std(results[i][:, 0].cpu().numpy())
            xmax = np.mean(results[i][:, 0].cpu().numpy()) + 2.5*np.std(results[i][:, 0].cpu().numpy())
            ymin = np.mean(results[i][:, 1].cpu().numpy()) - 2.5*np.std(results[i][:, 1].cpu().numpy())
            ymax = np.mean(results[i][:, 1].cpu().numpy()) + 2.5*np.std(results[i][:, 1].cpu().numpy())
            xlims = [xmin, xmax]
            ylims = [ymin, ymax]
        else:
            xlims = [-data_range,data_range]
            ylims = [-data_range, data_range]
            xmin = -data_range
            xmax = data_range
            ymin = -data_range
            ymax = data_range
        # Set up your x and y labels
        xlabel = '$\mathrm{a_1}$' + f'({i})'
        ylabel = '$\mathrm{a_2}$'+ f'({i})'

        x = results[i][:, 0].cpu().numpy()
        y = results[i][:, 1].cpu().numpy()

        # Set up the size of the figure
        # Define the locations for the axes
        # left, width = 0.12, 0.55
        # bottom, height = 0.12, 0.5
        # bottom_h = left_h = left+width+0.0
        
        rect_temperature = [left, bottom, width, height] # dimensions of temp plot
        # Make the three plots
        axTemperature = ax[n_inference_steps - i] # temperature plot
        # Make the 'main' temperature plot
        # Define the number of bins
        nxbins = 100
        nybins = 100
        nbins = 100
        
        xbins = linspace(start = xmin, stop = xmax, num = nxbins)
        ybins = linspace(start = ymin, stop = ymax, num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        
        # aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
        
        H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
        X = xcenter
        Y = ycenter
        Z = H
        axTemperature.scatter(action[:,0].cpu().detach().numpy(), action[:, 1].cpu().detach().numpy(), c='red', marker='x', s=100)

        # Plot the temperature data
        cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
            interpolation='nearest', origin='lower', aspect='auto'))
        
        if i == n_inference_steps:
            fig.tight_layout()
            axins = zoomed_inset_axes(axTemperature, zoom =window_zoom, loc=zoom_loc)
            axins.imshow(H, extent=[xmin,xmax,ymin,ymax],
            interpolation='nearest', origin='lower', aspect='auto')
            axins.scatter(action[:,0].cpu().detach().numpy(), action[:, 1].cpu().detach().numpy(), c='red', marker='x', s=100)

            xmin = np.mean(results[i][:, 0].cpu().numpy()) - 2*np.std(results[i][:, 0].cpu().numpy())
            xmax = np.mean(results[i][:, 0].cpu().numpy()) + 2*np.std(results[i][:, 0].cpu().numpy())
            ymin = np.mean(results[i][:, 1].cpu().numpy()) - 2*np.std(results[i][:, 1].cpu().numpy())
            ymax = np.mean(results[i][:, 1].cpu().numpy()) + 2*np.std(results[i][:, 1].cpu().numpy())
            x1, x2, y1, y2 = xmin, xmax, ymin, ymax
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            axins.set_xticks([])
            axins.set_yticks([])
            
            mark_inset(axTemperature, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # Plot the temperature plot contours
        contourcolor = 'white'
        # xcenter = np.mean(x)
        # ycenter = np.mean(y)
        ra = np.std(x)
        rb = np.std(y)
        ang = 0
        
        #Plot the axes labels
        # axTemperature.set_xlabel(xlabel,fontsize=10)
        # axTemperature.set_ylabel(ylabel,fontsize=10)
        axTemperature.set_xticklabels([])
        axTemperature.set_yticklabels([])
        axTemperature.set_xticks([])
        axTemperature.set_yticks([])
        #Make the tickmarks pretty
        '''ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(15)
            label.set_family('serif')
        
        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(15)
            label.set_family('serif')'''
        
        #Set up the plot limits
        axTemperature.set_xlim(xlims)
        axTemperature.set_ylim(ylims)
        
        #Set up the histogram bins
        # plt.hist2d(x, y, bins=(100, 100))
    file_name = os.path.join(file_path, f'histogram_v3_{n_inference_steps}_steps_idx_{random_idx}_' + noise_scheduler +'.png')
    plt.savefig(file_name)
    file_name = os.path.join(file_path, f'histogram_v3_{n_inference_steps}_steps_idx_{random_idx}_' + noise_scheduler +'.pdf')
    plt.savefig(file_name)
    plt.close()
    
    if make_gif:
        import imageio
        images = []
        # filenames = [os.path.join(model_store_path, f'histogram_{i}_v2_{3}_steps_idx_{randon_idx}.png') for i in range(0, 2)]
        for filename in filenames:
            images.append(imageio.imread(filename))
        for idx in range(0, 4):
            images.append(imageio.imread( filenames[-1]))
        imageio.mimsave(os.path.join(model_store_path, f'diffusion_v2_{n_inference_steps}_steps_idx_{random_idx}.gif'), images, duration=gif_duration)
        
        images = []
        # filenames = [os.path.join(model_store_path, f'histogram_{i}_v2_{3}_steps_idx_{randon_idx}.png') for i in range(0, 2)]
        for filename in filenames_2:
            images.append(imageio.imread(filename))
        for idx in range(0, 4):
            images.append(imageio.imread( filenames[-1]))
        imageio.mimsave(os.path.join(model_store_path, f'diffusion_{n_inference_steps}_steps_idx_{random_idx}.gif'), images, duration=gif_duration)
            
    print('done')

if __name__ == "__main__":
    # model_store_path = '/home/moritz/code/generative_learning_from_play/logs/multipath/runs/2023-01-15/19-57-28'
    model_store_path = '/home/moritz/code/generative_learning_from_play/logs/multipath/runs/2023-02-16/09-27-10'
    main(model_store_path)