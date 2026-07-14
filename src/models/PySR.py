from pysr import PySRRegressor

# Paper's loss: plain Huber, threshold=100 (LossFunctions.jl built-in via
# PySRRegressor's elementwise_loss -- no custom Julia needed).
HUBER_THRESHOLD = 100.0

ELEMENTWISE_LOSSES = {
    "huber": f"HuberLoss({HUBER_THRESHOLD})",
    "l2":    "L2DistLoss()",
    "l1":    "L1DistLoss()",
}

"""Hyper parameters in the loss function"""
# DELTA      = 5.0
# GAMMA      = 10.0
# GRAD_SCALE = 10.0
# w          = 3

# Kept for reference / future use -- not the current loss (see PySR()'s
# default loss="huber", matching the paper). Adds HSS-event-weighted
# asymmetric-bias and gradient-mismatch terms on top of a Huber base, using
# 03_run_sr_model.py's make_weights() per-sample alpha (dataset.weights).
CUSTOM_LOSS = """
function my_loss(tree, dataset::Dataset{T, L}, options)::L where {T, L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end

    y = dataset.y
    n = length(y)
    alpha = dataset.weights

    threshold  = T(100.0)
    delta      = T(5.0)
    gamma      = T(10.0)
    grad_scale = T(10.0)
    w          = 3

    total  = T(0)

    @inbounds for i in 1:n
        r  = prediction[i] - y[i]
        ar = abs(r)

        # Huber
        hub = ar <= threshold ? ar^2 / T(2) : threshold * (ar - threshold / T(2))

        a = (alpha === nothing) ? T(0) : alpha[i]

        if a > T(0)
            # Dead-zone asymmetric bias
            bias_diff = y[i] - prediction[i] - delta
            bias = bias_diff > T(0) ? bias_diff : T(0)

            # Gradient mismatch (3-hour window)
            tim = T(0)
            if i > w
                dyo = y[i] - y[i - w]
                dyp = prediction[i] - prediction[i - w]
                if dyo > gamma
                    lag_val = dyo - dyp
                    tim = lag_val > T(0) ? lag_val * grad_scale : T(0)
                end
            end

            total += (T(1) - a) * hub + a * (bias + tim)
        else
            total += hub
        end

    end

    return total / n
end
"""

def PySR(output_directory=None, run_id=None, loss="huber"):
    """
    loss : "huber" (default -- matches the paper: Huber loss, threshold=100),
           "l2", "l1", or "custom" (CUSTOM_LOSS above -- HSS-event-weighted,
           not the paper's actual loss).
    """
    if loss == "custom":
        loss_kwargs = dict(loss_function=CUSTOM_LOSS)
    elif loss in ELEMENTWISE_LOSSES:
        loss_kwargs = dict(elementwise_loss=ELEMENTWISE_LOSSES[loss])
    else:
        raise ValueError(f"Unknown loss {loss!r}; choose from 'custom', {list(ELEMENTWISE_LOSSES)}")

    model = PySRRegressor(
        population_size=100,
        ncycles_per_iteration=500,
        batching=False,

        maxsize=20,
        maxdepth=4,
        niterations=5000,

        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["log", "exp", "sqrt"],

        crossover_probability=0.12,
        parsimony=1e-4,
        optimize_probability=0.2,
        should_optimize_constants=True,

        **loss_kwargs,

        turbo=True,

        verbosity=1,
        progress=True,
        constraints={'^': (-1, 1)},

        random_state=42,
        deterministic=False,
        parallelism="multithreading",

        output_directory=output_directory,
        run_id=run_id,
    )
    return model