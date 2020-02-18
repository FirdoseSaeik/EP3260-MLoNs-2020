from __main__ import *

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def gradient_single(X, y, w, lmbda):
    l1 = np.matmul(X, w) * y
    yx = y * X

    return 2 * lmbda * w - sigmoid(-l1) * yx


def f_gradient(X, y, w):
    yX = y[:, np.newaxis] * X
    l1 = np.matmul(yX, w)
    return -sigmoid(-l1)[:, np.newaxis] * yX


def reg_gradient(w, lmbda):
    return 2 * lmbda * w


def gradient(X, y, w, lmbda):
    return reg_gradient(w, lmbda) + np.mean(f_gradient(X, y, w), axis=0)


def gradient_full(X, y, w, lmbda):
    return reg_gradient(w, lmbda) + f_gradient(X, y, w)


def cost(X, y, w, lmbda):

    l1 = np.matmul(X, w) * y
    return np.mean(np.log(1 + np.exp(-l1)), axis=0) + \
        lmbda * np.linalg.norm(w, 2)**2

def binary_classification_cost(X, y, w):
    return f1_score(y_true = y, y_pred = np.sign(np.matmul(X, w)), average='micro')

def ax_modifier(ax, legend_loc, ncol, xlabel, ylabel, title=None):
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    no_curves = len(ax.lines)
    ymin = min(ymin, min([min(ax.lines[i].get_ydata()) for i in range(no_curves)]))
    ymax = max(ymax, max([max(ax.lines[i].get_ydata()) for i in range(no_curves)]))
    xmax = max(xmax, max([max(ax.lines[i].get_xdata()) for i in range(no_curves)]))
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(np.arange(0, xmax+1, step=round(xmax/15)))
    ax.legend(ncol=ncol)
    if title is not None:
        ax.set_title(title)
    if not(np.isinf([ymin, ymax]).all()):  #if boundaries are defined
        ax.set_ylim((0.98*ymin, 1.02*ymax))


def random_sampler(N, batch=1, buffersize=10000):
    """
    A generator of random indices from 0 to N.
    params:
    N: upper bound of the indices
    batch: Number of indices to return per iteration
    buffersize: Number of numbers to generate per batch
                (this is only a computational nicety)
    """

    S = int(np.ceil(buffersize / batch))

    while True:
        buffer = np.random.randint(N, size=(S, batch))
        for i in range(S):
            yield buffer[i]


def GD(X, y, w, learning_rate=0.1, lmbda=0.01, iterations=1000):
    for iteration in range(iterations):
        grad = gradient(X, y, w, lmbda)
        w = w - learning_rate * grad

        yield w


def QGD(X, y, w, parameter_quantization_opt, grad_quantization_opt,
        learning_rate=0.1, lmbda=0.01, iterations=1000):
    """
    GD with quantization
    """
    no_samples, dimension = X.shape
    quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                    'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

    # The following code de-activates quantization of the initial gradient reporting at the outer-loop
    if isinstance(parameter_quantization_opt['no_bits'], int):
        quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        quantization['parameter']['center'] = np.tile([0], dimension)
        quantization['parameter']['radius'] = np.tile([parameter_quantization_opt['Range']], dimension)
    if isinstance(grad_quantization_opt['no_bits'], int):
        quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)
        quantization['gradient']['center'] = np.tile([0], dimension)
        quantization['gradient']['radius'] = np.tile([grad_quantization_opt['Range']], dimension)

    for iteration in range(iterations):
        quantized_grad = vector_quantization(original_vector = gradient(X, y, w, lmbda),
                                             quantization_radius = quantization['gradient']['radius'],
                                             grid_center = quantization['gradient']['center'],
                                             no_bits = quantization['gradient']['no_bits'])
        w = vector_quantization(original_vector = w - learning_rate * quantized_grad,
                                quantization_radius = quantization['parameter']['radius'],
                                grid_center = quantization['parameter']['center'],
                                no_bits = quantization['parameter']['no_bits'])

        yield w


def SGD(X, y, w, lmbda, learning_rate, batch_size=1):
    N, D = X.shape
    sampler = random_sampler(N, batch_size)

    for ix in sampler:
        grad = gradient(X[ix], y[ix], w, lmbda)
        w = w - learning_rate * grad
        yield w


def QSGD(X, y, w, lmbda, learning_rate, parameter_quantization_opt,
         grad_quantization_opt, batch_size=1):
    """
    SGD with quantization
    """
    no_samples, dimension = X.shape
    sampler = random_sampler(no_samples, batch_size)

    quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                    'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

    # The following code de-activates quantization of the initial gradient reporting at the outer-loop
    if isinstance(parameter_quantization_opt['no_bits'], int):
        quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        quantization['parameter']['center'] = np.tile([0], dimension)
        quantization['parameter']['radius'] = np.tile([parameter_quantization_opt['Range']], dimension)
    if isinstance(grad_quantization_opt['no_bits'], int):
        quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)
        quantization['gradient']['center'] = np.tile([0], dimension)
        quantization['gradient']['radius'] = np.tile([grad_quantization_opt['Range']], dimension)

    for ix in sampler:
        grad = gradient(X[ix], y[ix], w, lmbda)
        quantized_grad = vector_quantization(original_vector = grad,
                                             quantization_radius = quantization['gradient']['radius'],
                                             grid_center = quantization['gradient']['center'],
                                             no_bits = quantization['gradient']['no_bits'])
        w = vector_quantization(original_vector = w - learning_rate * quantized_grad,
                                quantization_radius = quantization['parameter']['radius'],
                                grid_center = quantization['parameter']['center'],
                                no_bits = quantization['parameter']['no_bits'])

        yield w


def SVRG(X, y, w, lmbda, learning_rate, epoch_size):
    """
    Stochastic variance reduced gradient
    """

    sampler = random_sampler(X.shape[0], epoch_size)

    for epoch in sampler:
        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        for ix in epoch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)
            w = w - learning_rate * (grad - full_grad[ix] + mean_grad)

        yield w

def M_SVRG(X, y, w, lmbda, learning_rate, epoch_size):
    """
    Stochastic variance reduced gradient
    """
    sampler = random_sampler(X.shape[0], epoch_size)

    for epoch in sampler:
        w_old = np.copy(w)
        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        for ix in epoch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)
            w = w - learning_rate * (grad - full_grad[ix] + mean_grad)

        mean_grad_new = np.mean(gradient_full(X, y, w, lmbda), axis=0)
        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            #print('------', w, w_old, np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2))
            print('---Switching to previous solution---  gradient norm: {:05.3f}, new gradient norm: {:05.3f}'.format(np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2)))
            w = np.copy(w_old)

        yield w


def SAG(X, y, w, lmbda, learning_rate, batch_size=1):
    N, D = X.shape
    P, = w.shape
    sampler = random_sampler(N, batch_size)

    grad = np.zeros((N, P))
    delta = np.zeros(P)
    non_zero_v = np.zeros(N)
    m = 0
    for ix in sampler:
        # update the number of seen examples m
        m -= np.sum(non_zero_v[ix], axis=0)
        non_zero_v[ix] = 1
        m += np.sum(non_zero_v[ix], axis=0)

        # update the sum of the gradient
        delta -= np.sum(grad[ix], axis=0)
        grad[ix] = f_gradient(X[ix], y[ix], w)
        delta += np.sum(grad[ix], axis=0)

        reg = reg_gradient(w, lmbda)

        w = w - learning_rate * (delta / m + reg)
        yield w


def QSAG(X, y, w, lmbda, learning_rate,parameter_quantization_opt,
         grad_quantization_opt, batch_size=1):
    """
    SAG with quantization
    """
    no_samples, dimension = X.shape
    P, = w.shape
    sampler = random_sampler(no_samples, batch_size)

    quantized_grad = np.zeros((no_samples, P))
    delta = np.zeros(P)
    non_zero_v = np.zeros(no_samples)
    m = 0

    quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                    'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

    # The following code de-activates quantization of the initial gradient reporting at the outer-loop
    if isinstance(parameter_quantization_opt['no_bits'], int):
        quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        quantization['parameter']['center'] = np.tile([0], dimension)
        quantization['parameter']['radius'] = np.tile([parameter_quantization_opt['Range']], dimension)
    if isinstance(grad_quantization_opt['no_bits'], int):
        quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)
        quantization['gradient']['center'] = np.tile([0], dimension)
        quantization['gradient']['radius'] = np.tile([grad_quantization_opt['Range']], dimension)

    for ix in sampler:
        # update the number of seen examples m
        m -= np.sum(non_zero_v[ix], axis=0)
        non_zero_v[ix] = 1
        m += np.sum(non_zero_v[ix], axis=0)

        # update the sum of the gradient
        delta -= np.sum(quantized_grad[ix], axis=0)
        quantized_grad[ix] = vector_quantization(original_vector = f_gradient(X[ix], y[ix], w)[0],
                                                 quantization_radius = quantization['gradient']['radius'],
                                                 grid_center = quantization['gradient']['center'],
                                                 no_bits = quantization['gradient']['no_bits'])

        delta += np.sum(quantized_grad[ix], axis=0)
        reg = reg_gradient(w, lmbda)

        w = vector_quantization(original_vector = w - learning_rate * (delta / m + reg),
                                quantization_radius = quantization['parameter']['radius'],
                                grid_center = quantization['parameter']['center'],
                                no_bits = quantization['parameter']['no_bits'])
        yield w


def initialize_w(N):
    return np.random.randn(N)


def loss(X, y, w, lmbda):
    objective_loss = cost(X, y, w, lmbda)
    f1_score = binary_classification_cost(X, np.sign(y), w)
    return objective_loss, f1_score


def iterate(opt, X_train, y_train, X_test, y_test, w_0,
            lmbda, iterations=100, inner=1, name="NoName", printout=True):
    """
    This function takes an optimizer and returns a loss history for the
    training and test sets.
    """

    loss_hist_train, train_f1_score = loss(X_train, y_train, w_0, lmbda)
    loss_hist_test, test_f1_score = loss(X_test, y_test, w_0, lmbda)
    mean_grad_norm = np.linalg.norm(np.mean(gradient_full(X_train, y_train, w_0, lmbda), axis=0), 2)

    ws = [w_0]
    clock = [0]

    start = time.time()
    for iteration in range(iterations):
        #print("\n ------Iteration {} is started-----".format(iteration))
        for _ in range(inner):
            w = next(opt)
        clock.append(time.time() - start)
        ws.append(w)

    #for iteration, w in enumerate(ws):
        train_loss, train_f1_score_new = loss(X_train, y_train, w, lmbda)
        loss_hist_train = np.append(loss_hist_train, train_loss)
        train_f1_score = np.append(train_f1_score, train_f1_score_new)

        test_loss, test_f1_score_new = loss(X_test, y_test, w, lmbda)
        loss_hist_test = np.append(loss_hist_test, test_loss)
        test_f1_score = np.append(test_f1_score, test_f1_score_new)
        grad_norm_new = np.linalg.norm(np.mean(gradient_full(X_train, y_train, w, lmbda), axis=0), 2)
        mean_grad_norm = np.append(mean_grad_norm, grad_norm_new)

        if printout:
            print('{}; Iter = {:02}; Objective(train) = {:05.3f}; Objective(test) = {:05.3f}; F1score(train) = {:05.3f}; F1score(test) = {:05.3f}'.format(name, iteration, train_loss, test_loss, train_f1_score_new, test_f1_score_new))
        #print('Current solution = {}'.format(w.round(decimals=3)))
        sys.stdout.flush()

    return ws[-1], loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm


def iter_latency(algorithm_name, hyper_parameters, rate_model):
    """
    rate_model['p']: rate of a point-to-point channel
    rate_model['m']: rate of a multiple access channel
    latency: latency of running one iteration. In teh case SVRG family, it is for running one outer-loop.
    """
    if algorithm_name in ['GD', 'Q-GD']:
        no_bits_uplink = hyper_parameters['dimension']*hyper_parameters['grad_quantization_opt']['no_bits'] + hyper_parameters['grad_communication_overhead']
        no_bits_downlink = hyper_parameters['dimension']*hyper_parameters['parameter_quantization_opt']['no_bits'] + hyper_parameters['param_communication_overhead']
        latency = no_bits_downlink/rate_model['p'] + no_bits_uplink/rate_model['m']


    elif algorithm_name in ['SGD', 'Q-SGD', 'SAG', 'Q-SAG']:
        no_bits_uplink = hyper_parameters['dimension']*hyper_parameters['grad_quantization_opt']['no_bits'] + hyper_parameters['grad_communication_overhead']
        no_bits_downlink = hyper_parameters['dimension']*hyper_parameters['parameter_quantization_opt']['no_bits'] + hyper_parameters['param_communication_overhead']
        latency = (no_bits_downlink+no_bits_uplink)/rate_model['p']


    elif algorithm_name in ['SVRG', 'M-SVRG', 'QM-SVRG-F', 'QM-SVRG-A', 'QM-SVRG-F-plus', 'QM-SVRG-A-plus']:
        no_bits_uplink = hyper_parameters['dimension']*hyper_parameters['grad_quantization_opt']['no_bits'] + hyper_parameters['grad_communication_overhead']
        no_bits_downlink = hyper_parameters['dimension']*hyper_parameters['parameter_quantization_opt']['no_bits'] + hyper_parameters['param_communication_overhead']
        inner_loop_latency = hyper_parameters['SVRG_epoch_size']*(no_bits_uplink+no_bits_downlink)/rate_model['p']
        outer_loop_latency = no_bits_uplink/rate_model['m']
        latency = inner_loop_latency + outer_loop_latency

    return latency




def quantize(original_value, quantization_radius, grid_center, no_bits):
    """
    You should run this for every entry of a vector
    """
    no_grid_points = np.array(2**no_bits, dtype=np.int64)
    grid_points = grid_center + np.linspace(-quantization_radius, quantization_radius, num=no_grid_points,
                                            endpoint=True, dtype=np.float64)

    xxx = np.abs(grid_points - original_value)
    quantized_value = grid_points[(np.abs(grid_points - original_value)).argmin()]


    return quantized_value


def vector_quantization(original_vector, quantization_radius, grid_center, no_bits):
    dimension = len(original_vector)
    quantized_vector = [quantize(original_vector[i], quantization_radius[i], grid_center[i], no_bits[i]) for i in range(dimension)]
    quantized_vector = np.array(quantized_vector, dtype=np.float64)

    return quantized_vector

def QM_SVRG_F(X, y, w, lmbda, learning_rate, epoch_size,
              parameter_quantization_opt, grad_quantization_opt):
    """
    Stochastic variance reduced gradient with fixed quantization grids
    """

    no_samples, dimension = X.shape
    sampler = random_sampler(no_samples, epoch_size)

    for epoch in sampler:
        w_old = np.copy(w)
        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)
        quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                        'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

        # The following code de-activates quantization of the initial gradient reporting at the outer-loop
        if isinstance(parameter_quantization_opt['no_bits'], int):
            quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
            quantization['parameter']['center'] = np.tile([0], dimension)
            quantization['parameter']['radius'] = np.tile([parameter_quantization_opt['Range']], dimension)

        if isinstance(grad_quantization_opt['no_bits'], int):
            quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)
            quantization['gradient']['center'] = np.tile([0], dimension)
            quantization['gradient']['radius'] = np.tile([grad_quantization_opt['Range']], dimension)

        for ix in epoch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)

            quantized_grad = vector_quantization(original_vector=grad,
                                                 quantization_radius=quantization['gradient']['radius'],
                                                 grid_center=quantization['gradient']['center'],
                                                 no_bits=quantization['gradient']['no_bits'])

            w = w - learning_rate * (quantized_grad - full_grad[ix] + mean_grad)
            w = vector_quantization(original_vector=w,
                                    quantization_radius=quantization['parameter']['radius'],
                                    grid_center=quantization['parameter']['center'],
                                    no_bits=quantization['parameter']['no_bits'])

        mean_grad_new = np.mean(gradient_full(X, y, w, lmbda), axis=0)

        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            #print('\n-NoU-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w, '\n')
            w = np.copy(w_old)
        #else:
        #    print('-U-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w)


        yield w


def QM_SVRG_A(X, y, w, lmbda, learning_rate, epoch_size,
              parameter_quantization_opt, grad_quantization_opt, strong_convexity_param):

    """
    Stochastic variance reduced gradient with fixed quantization grids
    """

    no_samples, dimension = X_train.shape
    sampler = random_sampler(no_samples, epoch_size)

    for batch in sampler:
        w_old = np.copy(w)
        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                        'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

        # The following code de-activates quantization of the initial gradient reporting at the outer-loop
        if isinstance(parameter_quantization_opt['no_bits'], int):
            quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        if isinstance(grad_quantization_opt['no_bits'], int):
            quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)


        quantization['gradient']['center'] = mean_grad
        quantization['gradient']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2), dimension)
        quantization['parameter']['center'] = w_old
        quantization['parameter']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2) / strong_convexity_param, dimension)

        #print('-- Grid_center is {} \n -- Quantization radius is {}'.format(grid_center.round(decimals=3), quantization_radius[0].round(decimals=3)))
        #        print('grads are ', np.mean(full_grad, axis=0), '--norm is ', np.linalg.norm(np.mean(full_grad, axis=0)), '--strong_convexity_param is ', strong_convexity_param)

        for ix in batch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)
            quantized_grad = vector_quantization(original_vector=grad,
                                                 quantization_radius=quantization['gradient']['radius'],
                                                 grid_center=quantization['gradient']['center'],
                                                 no_bits=quantization['gradient']['no_bits'])

            w = w - learning_rate * (quantized_grad - full_grad[ix] + mean_grad)
            w = vector_quantization(original_vector=w,
                                    quantization_radius=quantization['parameter']['radius'],
                                    grid_center=quantization['parameter']['center'],
                                    no_bits=quantization['parameter']['no_bits'])

        mean_grad_new = np.mean(gradient_full(X, y, w, lmbda), axis=0)
        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            #print('\n-NoU-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w, '\n')
            w = np.copy(w_old)

        #else:
            #print('-U-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w)

        yield w



    '''
    # The following code activates quantization of the initial gradient reporting at the outer-loop

    local_datasets = np.array([i*np.floor(no_samples/(no_nodes))   for i in range(no_nodes)], dtype=np.int64)
    local_datasets = np.append(local_datasets, [no_samples])
    for batch in sampler:
        full_grad = gradient_full(X_train, y_train, w, hyper_parameters['lambda'])
        grid_center = np.mean(full_grad, axis=0)
        quantization_radius = np.tile([2 * smoothness_param * np.linalg.norm(np.mean(full_grad, axis=0)) / strong_convexity_param], dimension)
        print('-- Grid_center is {} \n -- Quantization radius is {}'.format(grid_center.round(decimals=3), quantization_radius[0].round(decimals=3)))
        #        print('grads are ', np.mean(full_grad, axis=0), '--norm is ', np.linalg.norm(np.mean(full_grad, axis=0)), '--strong_convexity_param is ', strong_convexity_param)
        if isinstance(no_bits, int):
            no_bits_vect = np.tile([no_bits], dimension)
        sum_quantized_grad = np.zeros(dimension, dtype=np.float64)
        for i in range(no_nodes):
            full_grad_node = full_grad[local_datasets[i]:local_datasets[i+1]]
            mean_grad_node = np.mean(full_grad_node, axis=0)

            # Quantization step
            quantized_mean_grad_node = vector_quantization(original_vector = mean_grad_node,
                                                           quantization_radius = quantization_radius,
                                                           grid_center = grid_center, no_bits = no_bits_vect)
            sum_quantized_grad = np.add(sum_quantized_grad, quantized_mean_grad_node)

        mean_quantized_grad = sum_quantized_grad/no_nodes

        for ix in batch:
            grad = gradient_single(X_train[ix], y_train[ix], w, hyper_parameters['lambda'])
            quantized_grad = vector_quantization(original_vector = grad, quantization_radius = quantization_radius,
                                                 grid_center = grid_center, no_bits = no_bits_vect)
            quantized_full_grad_ix = vector_quantization(original_vector = full_grad[ix], quantization_radius = quantization_radius,
                                                         grid_center = grid_center, no_bits = no_bits_vect)
            w = w - hyper_parameters['learning_rate'] * (quantized_grad - quantized_full_grad_ix + mean_quantized_grad)
        yield w
    '''





def QM_SVRG_F_plus(X, y, w, lmbda, learning_rate, epoch_size,
              parameter_quantization_opt, grad_quantization_opt):
    """
    Stochastic variance reduced gradient with fixed quantization grids
    """

    no_samples, dimension = X.shape
    sampler = random_sampler(no_samples, epoch_size)

    for epoch in sampler:
        w_old = np.copy(w)
        quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                        'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

        # The following code de-activates quantization of the initial gradient reporting at the outer-loop
        if isinstance(parameter_quantization_opt['no_bits'], int):
            quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
            quantization['parameter']['center'] = np.tile([0], dimension)
            quantization['parameter']['radius'] = np.tile([parameter_quantization_opt['Range']], dimension)

        if isinstance(grad_quantization_opt['no_bits'], int):
            quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)
            quantization['gradient']['center'] = np.tile([0], dimension)
            quantization['gradient']['radius'] = np.tile([grad_quantization_opt['Range']], dimension)

        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        for ix in epoch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)

            quantized_grad = vector_quantization(original_vector=grad,
                                                 quantization_radius=quantization['gradient']['radius'],
                                                 grid_center=quantization['gradient']['center'],
                                                 no_bits=quantization['gradient']['no_bits'])

            full_grad_quantized = vector_quantization(original_vector=full_grad[ix],
                                                      quantization_radius=quantization['gradient']['radius'],
                                                      grid_center=quantization['gradient']['center'],
                                                      no_bits=quantization['gradient']['no_bits'])

            w = w - learning_rate * (quantized_grad - full_grad_quantized + mean_grad)
            w = vector_quantization(original_vector=w,
                                    quantization_radius=quantization['parameter']['radius'],
                                    grid_center=quantization['parameter']['center'],
                                    no_bits=quantization['parameter']['no_bits'])

        mean_grad_new = np.mean(gradient_full(X, y, w, lmbda), axis=0)
        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            #print('\n-NoU-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w, '\n')
            w = np.copy(w_old)

        yield w


def QM_SVRG_A_plus(X, y, w, lmbda, learning_rate, epoch_size,
              parameter_quantization_opt, grad_quantization_opt, strong_convexity_param):

    """
    Stochastic variance reduced gradient with fixed quantization grids
    """

    no_samples, dimension = X_train.shape
    sampler = random_sampler(no_samples, epoch_size)

    for batch in sampler:
        w_old = np.copy(w)

        quantization = {'parameter':{'no_bits':{}, 'center':{}, 'radius':{}},
                        'gradient':{'no_bits':{}, 'center':{}, 'radius':{}}}

        # The following code de-activates quantization of the initial gradient reporting at the outer-loop
        if isinstance(parameter_quantization_opt['no_bits'], int):
            quantization['parameter']['no_bits'] = np.tile([parameter_quantization_opt['no_bits']], dimension)
        if isinstance(grad_quantization_opt['no_bits'], int):
            quantization['gradient']['no_bits'] = np.tile([grad_quantization_opt['no_bits']], dimension)


        full_grad = gradient_full(X, y, w, lmbda)
        mean_grad = np.mean(full_grad, axis=0)

        quantization['gradient']['center'] = mean_grad
        quantization['gradient']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2), dimension)
        quantization['parameter']['center'] = w_old
        quantization['parameter']['radius'] = np.tile(2 * np.linalg.norm(mean_grad, 2) / strong_convexity_param, dimension)

        #print('-- Grid_center is {} \n -- Quantization radius is {}'.format(grid_center.round(decimals=3), quantization_radius[0].round(decimals=3)))
        #        print('grads are ', np.mean(full_grad, axis=0), '--norm is ', np.linalg.norm(np.mean(full_grad, axis=0)), '--strong_convexity_param is ', strong_convexity_param)


        for ix in batch:
            grad = gradient_single(X[ix], y[ix], w, lmbda)
            quantized_grad = vector_quantization(original_vector = grad,
                                                 quantization_radius = quantization['gradient']['radius'],
                                                 grid_center = quantization['gradient']['center'],
                                                 no_bits = quantization['gradient']['no_bits'])

            full_grad_quantized = vector_quantization(original_vector = full_grad[ix],
                                                      quantization_radius = quantization['gradient']['radius'],
                                                      grid_center = quantization['gradient']['center'],
                                                      no_bits = quantization['gradient']['no_bits'])

            w = w - learning_rate * (quantized_grad - full_grad_quantized + mean_grad)
            w = vector_quantization(original_vector = w,
                                    quantization_radius = quantization['parameter']['radius'],
                                    grid_center = quantization['parameter']['center'],
                                    no_bits = quantization['parameter']['no_bits'])

        mean_grad_new = np.mean(gradient_full(X, y, w, lmbda), axis=0)
        if np.linalg.norm(mean_grad, 2) <= np.linalg.norm(mean_grad_new, 2):
            #print('\n-NoU-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w, '\n')
            w = np.copy(w_old)

        #else:
            #print('-U-', np.linalg.norm(mean_grad, 2), np.linalg.norm(mean_grad_new, 2), w_old, w)

        yield w


def T_LB_FixedGrid(mu, alpha, L, sigma_0):
    return 1/(mu* alpha * (sigma_0 - 2*L*alpha*sigma_0 - 2*L*alpha))


def T_LB_AdaptiveGrid(mu, alpha, L, b, d, sigma_0):
    return max(1, 1/(mu*alpha*(sigma_0 - 2*L*alpha*sigma_0 - 2*L*alpha) - (4*L*d*(1+4*(mu**2)* (alpha**2)))/((2**(b/d)-1)**2)))


def alpha_UB_FixedGrid(L, sigma_0):
    return sigma_0/(2*L*(1+sigma_0))


def alpha_UB_AdaptiveGrid(L, sigma_0):
    return sigma_0/(2*L*(1+sigma_0))


def noBitsPerDimension_LB(d, L, mu, alpha, sigma_0):
    return np.ceil(np.log2(1+np.sqrt((4*L*d*(1+4*(mu**2)*(alpha**2)))/((mu**2)*alpha*(sigma_0 -2*L*alpha*sigma_0 -2*L*alpha)))))


def optimizer_no_quantization_binary_classification(target_dataset, X_train, y_train,
                                                    X_test, y_test, w_0,
                                                    hyper_parameters, multi_class=False, printout=True):
    optimizers = [
            {
                    "opt": SGD(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                               learning_rate=hyper_parameters['learning_rate'], batch_size=1),
                    "name": "SGD",
                    "inner": 1
            },
            {
                    "opt": SAG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                               learning_rate=hyper_parameters['learning_rate']),
                    "name": "SAG",
                    "inner": 1
            },
            {
                    "opt": SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate'],
                                epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "SVRG",
                    "inner": 1
            },
            {
                    "opt": M_SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                  learning_rate=hyper_parameters['learning_rate'],
                                  epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "M-SVRG",
                    "inner": 1
            },
            {
                    "opt": GD(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                              learning_rate=hyper_parameters['learning_rate'],
                              iterations = hyper_parameters['iterations']),
                    "name": "GD",
                    "inner": 1
            },
    ]

    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score': {}, 'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}

    fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train, y_train, X_test, y_test, w_0,
                lmbda=hyper_parameters['lambda'],
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)
        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        color = next(ax[0]._get_lines.prop_cycler)['color']
        iterations_axis = range(0, hyper_parameters['iterations'] + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
               label="Train loss ({})".format(opt['name']), linestyle="-", color=color)

        ax[0].plot(iterations_axis, loss_hist_test,
               label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

        ax[1].plot(iterations_axis, mean_grad_norm,
               label="Training ({})".format(opt['name']), linestyle="-", color=color)

        #ax[2].plot(iterations_axis, train_f1_score,
        #       label="Train F1 score ({})".format(opt['name']), linestyle="-", color=color)

        ax[2].plot(iterations_axis, test_f1_score,
               label="{}".format(opt['name']), linestyle="-", color=color)

        #ax[3].plot(clock, loss_hist_train,
        #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
        #ax[3].plot(clock, loss_hist_test,
        #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

    ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Loss",
                title="Performance Comparison of various algorithms")
    ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Gradient norm (training)")
    ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel="Iteration", ylabel="F1 score (test)")
    #ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

    if multi_class:
        fig_name = './TestResults/noQuant_allAlg_'+target_dataset+'_Class'+str(y_train[0])
    else:
        fig_name = './TestResults/noQuant_allAlg_'+target_dataset+'_BinaryClassification'

    plt.savefig(fig_name+'.png')
    plt.savefig(fig_name+'.pdf')
    matplotlib2tikz.save(fig_name+'.tex')

    return outputs


def optimizer_quantization_binary_classification(target_dataset, X_train, y_train,
                                                 X_test, y_test, w_0,
                                                 hyper_parameters, multi_class=False, plot=True, printout=True):
    optimizers = [
            {
                    "opt": GD(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                              learning_rate=hyper_parameters['learning_rate'],
                              iterations = hyper_parameters['iterations']),
                    "name": "GD",
                    "inner": 1
            },
            {
                    "opt": QGD(X=X_train, y=y_train, w=w_0,
                               parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                               grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                               lmbda=hyper_parameters['lambda'],
                               learning_rate=hyper_parameters['learning_rate'],
                               iterations = hyper_parameters['iterations']),
                    "name": "Q-GD",
                    "inner": 1
            },
            {
                    "opt": QSGD(X=X_train, y=y_train, w=w_0,
                                parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate'], batch_size=1),
                    "name": "Q-SGD",
                    "inner": 1
            },
            {
                    "opt": QSAG(X=X_train, y=y_train, w=w_0,
                                parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate']),
                    "name": "Q-SAG",
                    "inner": 1
            },
            {
                    "opt": SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate'],
                                epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "SVRG",
                    "inner": 1
            },
            {
                    "opt": M_SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                  learning_rate=hyper_parameters['learning_rate'],
                                  epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "M-SVRG",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_F(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                     learning_rate=hyper_parameters['learning_rate'],
                                     epoch_size=hyper_parameters['SVRG_epoch_size'],
                                     parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                     grad_quantization_opt=hyper_parameters['grad_quantization_opt']),
                    "name": "QM-SVRG-F",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_A(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                     learning_rate=hyper_parameters['learning_rate'],
                                     epoch_size=hyper_parameters['SVRG_epoch_size'],
                                     parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                     grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                     strong_convexity_param=hyper_parameters['strong_convexity_param']),
                    "name": "QM-SVRG-A",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_F_plus(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                          learning_rate=hyper_parameters['learning_rate'],
                                          epoch_size=hyper_parameters['SVRG_epoch_size'],
                                          parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                          grad_quantization_opt=hyper_parameters['grad_quantization_opt']),
                    "name": "QM-SVRG-F-plus",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_A_plus(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                          learning_rate=hyper_parameters['learning_rate'],
                                          epoch_size=hyper_parameters['SVRG_epoch_size'],
                                          parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                          grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                          strong_convexity_param=hyper_parameters['strong_convexity_param']),
                    "name": "QM-SVRG-A-plus",
                    "inner": 1
            },
    ]

    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score':  {}, 'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}
    if plot:
        fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train, y_train, X_test, y_test, w_0,
                lmbda=hyper_parameters['lambda'],
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)
        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        if plot:
            color = next(ax[0]._get_lines.prop_cycler)['color']
            iterations_axis = range(0, hyper_parameters['iterations'] + 1)
            ax[0].plot(iterations_axis, loss_hist_train,
                       label="{}".format(opt['name']), linestyle="-", color=color)

            # ax[0].plot(iterations_axis, loss_hist_test,
            #        label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

            ax[1].plot(iterations_axis, mean_grad_norm,
                       label="{}".format(opt['name']), linestyle="-", color=color)

            #ax[2].plot(iterations_axis, train_f1_score,
            #       label="Train F1 score ({})".format(opt['name']), linestyle="-", color=color)

            ax[2].plot(iterations_axis, test_f1_score,
                       label="{}".format(opt['name']), linestyle="-", color=color)

            #ax[3].plot(clock, loss_hist_train,
            #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
            #ax[3].plot(clock, loss_hist_test,
            #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)


    if plot:
        ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="", ylabel="Loss (training)")
        ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="", ylabel="Gradient norm (training)")
        ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel='Iteration', ylabel="F1 score (test)")
        #ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

        if multi_class:
            fig_name = './TestResults/P_'+str(hyper_parameters['parameter_quantization_opt']['no_bits'])+'_Q_'+str(hyper_parameters['grad_quantization_opt']['no_bits'])+'_withQuant_'+target_dataset+'_Class'+str(y_train[0])
        else:
            fig_name = './TestResults/P_'+str(hyper_parameters['parameter_quantization_opt']['no_bits'])+'_Q_'+str(hyper_parameters['grad_quantization_opt']['no_bits'])+'_withQuant_'+target_dataset+'_BinaryClassification'

        plt.savefig(fig_name+'.png')
        plt.savefig(fig_name+'.pdf')
        matplotlib2tikz.save(str(fig_name+'.tex'))

    return outputs


def optimizer_quantization_multiclass_classification(target_dataset, X_train, y_train,
                                                     X_test, y_test, w_0,
                                                     hyper_parameters, class_idx, printout=True):
    optimizers = [
            {
                    "opt": GD(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                              learning_rate=hyper_parameters['learning_rate'],
                              iterations = hyper_parameters['iterations']),
                    "name": "GD",
                    "inner": 1
            },
            {
                    "opt": QGD(X=X_train, y=y_train, w=w_0,
                               parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                               grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                               lmbda=hyper_parameters['lambda'],
                               learning_rate=hyper_parameters['learning_rate'],
                               iterations = hyper_parameters['iterations']),
                    "name": "Q-GD",
                    "inner": 1
            },
            {
                    "opt": QSGD(X=X_train, y=y_train, w=w_0,
                                parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate'], batch_size=1),
                    "name": "Q-SGD",
                    "inner": 1
            },
            {
                    "opt": QSAG(X=X_train, y=y_train, w=w_0,
                                parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate']),
                    "name": "Q-SAG",
                    "inner": 1
            },
            {
                    "opt": SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                learning_rate=hyper_parameters['learning_rate'],
                                epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "SVRG",
                    "inner": 1
            },
            {
                    "opt": M_SVRG(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                  learning_rate=hyper_parameters['learning_rate'],
                                  epoch_size=hyper_parameters['SVRG_epoch_size']),
                    "name": "M-SVRG",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_F(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                     learning_rate=hyper_parameters['learning_rate'],
                                     epoch_size=hyper_parameters['SVRG_epoch_size'],
                                     parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                     grad_quantization_opt=hyper_parameters['grad_quantization_opt']),
                    "name": "QM-SVRG-F",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_A(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                     learning_rate=hyper_parameters['learning_rate'],
                                     epoch_size=hyper_parameters['SVRG_epoch_size'],
                                     parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                     grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                     strong_convexity_param=hyper_parameters['strong_convexity_param']),
                    "name": "QM-SVRG-A",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_F_plus(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                          learning_rate=hyper_parameters['learning_rate'],
                                          epoch_size=hyper_parameters['SVRG_epoch_size'],
                                          parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                          grad_quantization_opt=hyper_parameters['grad_quantization_opt']),
                    "name": "QM-SVRG-F-plus",
                    "inner": 1
            },
            {
                    "opt": QM_SVRG_A_plus(X=X_train, y=y_train, w=w_0, lmbda=hyper_parameters['lambda'],
                                          learning_rate=hyper_parameters['learning_rate'],
                                          epoch_size=hyper_parameters['SVRG_epoch_size'],
                                          parameter_quantization_opt=hyper_parameters['parameter_quantization_opt'],
                                          grad_quantization_opt=hyper_parameters['grad_quantization_opt'],
                                          strong_convexity_param=hyper_parameters['strong_convexity_param']),
                    "name": "QM-SVRG-A-plus",
                    "inner": 1
            },
    ]
    outputs = {optimizers[i]['name']: {'optimal_parameter': {}, 'training_loss': {}, 'training_f1_score': {},
                                       'training_mean_grad_norm': {}}
               for i in np.arange(len(optimizers))}

    fig, ax = plt.subplots(3, 1, figsize=(13, 12))

    for opt in optimizers:

        w, loss_hist_train, loss_hist_test, train_f1_score, test_f1_score, clock, mean_grad_norm = iterate(
                opt['opt'],
                X_train, y_train, X_test, y_test, w_0,
                lmbda=hyper_parameters['lambda'],
                iterations=hyper_parameters['iterations'], inner=opt['inner'],
                name=opt['name'], printout=printout)

        outputs[opt['name']]['optimal_parameter'] = w
        outputs[opt['name']]['training_loss'] = loss_hist_train
        outputs[opt['name']]['training_f1_score'] = train_f1_score
        outputs[opt['name']]['training_mean_grad_norm'] = mean_grad_norm

        color = next(ax[0]._get_lines.prop_cycler)['color']

        iterations_axis = range(0, hyper_parameters['iterations'] + 1)
        ax[0].plot(iterations_axis, loss_hist_train,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[1].plot(iterations_axis, mean_grad_norm,
               label="{}".format(opt['name']), linestyle="-", color=color)

        ax[2].plot(iterations_axis, test_f1_score,
               label="{}".format(opt['name']), linestyle="-", color=color)

        #ax[3].plot(clock, loss_hist_train,
        #           label="Train loss ({})".format(opt['name']), linestyle="-", color=color)
        #ax[3].plot(clock, loss_hist_test,
        #           label="Test loss ({})".format(opt['name']), linestyle="--", color=color)

    ax_modifier(ax=ax[0], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Loss (training)",
                title="Performance Comparison of various algorithms")
    ax_modifier(ax=ax[1], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Gradient norm (training)")
    ax_modifier(ax=ax[2], legend_loc="lower right", ncol=2, xlabel="Iteration", ylabel="F1 score (test)")
    #ax_modifier(ax=ax[3], legend_loc="upper right", xlabel="Time in seconds", ylabel="Loss")

    fig_name = './TestResults/P_'+str(hyper_parameters['parameter_quantization_opt']['no_bits'])+'_Q_'+str(hyper_parameters['grad_quantization_opt']['no_bits'])+'_withQuant_'+target_dataset+'_Class'+str(class_idx)

    plt.savefig(fig_name+'.png')
    plt.savefig(fig_name+'.pdf')
    matplotlib2tikz.save(fig_name+'.tex')

    return outputs


def latency_evaluation(target_dataset, optimizers, hyper_parameters, no_class, plot=True):

    no_scenarios = len(hyper_parameters['rate_model'])
    if plot:
        fig, ax = plt.subplots(no_scenarios, 1, figsize=(13, 16))
    i = 0
    latency = {'scenario_'+str(i+1): {j: {} for j in optimizers} for i in range(len(hyper_parameters['rate_model']))}
    iterations_axis = range(0, hyper_parameters['iterations'] + 1)
    for rate_model in hyper_parameters['rate_model']:
        for algorithm_name in optimizers:
            # with one-vs-all, the time complexity becomes no_class times higher than binary, since we need to solve the same
            # problem no_class times.

            if no_class == 2:
                latency_per_iter = iter_latency(algorithm_name=algorithm_name, hyper_parameters=hyper_parameters,
                                                rate_model=rate_model)
            else:
                latency_per_iter = no_class*iter_latency(algorithm_name=algorithm_name,
                                                         hyper_parameters=hyper_parameters, rate_model=rate_model)

            # append [0] at the beginning
            latency_array = np.array([0] + [latency_per_iter for i in range(hyper_parameters['iterations'])])
            #latency_array = np.cumsum([0, *latency_array])
            latency['scenario_'+str(i+1)][algorithm_name] = np.cumsum(latency_array)

            if plot:
                color = next(ax[0]._get_lines.prop_cycler)['color']
                ax[i].plot(iterations_axis, latency['scenario_'+str(i+1)][algorithm_name],
                       label="{} (scenario {})".format(algorithm_name, i+1), linestyle="-", color=color)

        if plot:
            ax_modifier(ax=ax[i], legend_loc="upper right", ncol=2, xlabel="Iteration", ylabel="Time complexity [s]")
        i += 1

    if plot:
        fig_name = './TestResults/P_'+str(hyper_parameters['parameter_quantization_opt']['no_bits'])+'_Q_'+str(hyper_parameters['grad_quantization_opt']['no_bits'])+'_TimeComplexity_'+target_dataset
        plt.savefig(fig_name+'.png')
        plt.savefig(fig_name+'.pdf')
        #matplotlib2tikz.save(fig_name+'.tex')

    return latency


def time_complexy_vs_bits(outputs, grad_threshold, plot=True):
    no_bits_sweep = outputs['no_bits_sweep']

    rate_scenarios = [rate_scenario for rate_scenario in outputs['time_complexity'][0].keys()]
    optimizers = [name for name in outputs['solution'][0].keys()]
    convergence_time = {rate_scenario: {name: [] for name in optimizers} for rate_scenario in rate_scenarios}

    if plot:
        fig, ax = plt.subplots(len(rate_scenarios)+1, 1, figsize=(13, 16))
        i = 0

    for rate_scenario in rate_scenarios:
        for algorithm_name in optimizers:
            for no_bits_idx in range(len(no_bits_sweep)):
                grad_norm = outputs['solution'][no_bits_idx][algorithm_name]['training_mean_grad_norm']
                tmp = [x[0] for x in enumerate(grad_norm) if x[1] <= grad_threshold]
                if tmp:
                    convergence_iter_number = tmp[0]
                else:
                    convergence_iter_number = np.nan

                if np.isnan(convergence_iter_number):
                    convergence_time_new = np.nan
                else:
                    convergence_time_new = outputs['time_complexity'][no_bits_idx][rate_scenario][algorithm_name][convergence_iter_number]
                convergence_time[rate_scenario][algorithm_name].append(convergence_time_new)
            if plot:
                color = next(ax[0]._get_lines.prop_cycler)['color']
                ax[i].plot(no_bits_sweep, convergence_time[rate_scenario][algorithm_name],
                           label="{} ({})".format(algorithm_name, rate_scenario), linestyle="-", color=color)
        if plot:
            ax_modifier(ax=ax[i], legend_loc="upper right", ncol=2, xlabel="Number of bits", ylabel="Time complexity [s]")
            i += 1

    return convergence_time