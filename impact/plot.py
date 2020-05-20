from pmd_beamphysics.units import nice_array
import matplotlib.pyplot as plt

def plot_stat(impact_object, y='sigma_x', x='mean_z', nice=True):
    """
    Plots stat output of key y vs key
    
    If particles have the same stat key, these will also be plotted.
    
    If nice, a nice SI prefix and scaling will be used to make the numbers reasonably sized.
    
    """
    I = impact_object # convenience
    fig, ax = plt.subplots()

    units1 = str(I.units(x))
    units2 = str(I.units(y))

    X = I.stat(x)
    Y = I.stat(y)
    
    if nice:
        X, f1, prefix1 = nice_array(X)
        Y, f2, prefix2 = nice_array(Y)
        units1  = prefix1+units1
        units2  = prefix2+units2
    else:
        f1 = 1
        f2 = 2    
    ax.set_xlabel(x+f' ({units1})')
    ax.set_ylabel(y+f' ({units2})')
    
    # line plot
    plt.plot(X, Y)
    
    try:
        ax.scatter(
            [I.particles[name][x]/f1 for name in I.particles],
            [I.particles[name][y]/f2 for name in I.particles],  color='red')  
    except:
        pass
    
    #return fig
    
