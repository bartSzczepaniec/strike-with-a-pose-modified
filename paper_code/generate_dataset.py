import os
import math
from datetime import datetime

import cma
import imageio
import logging

from PIL import ImageDraw
from optimizer_settings import *
from renderer import Renderer
from strike_utils import *


def get_start_params(z_samples=30, random_samples=10):
    """Find good starting parameters.

    :param z_samples:
    :param random_samples:
    :return:
    """
    best_loss = float("inf")
    best_params = {}
    z_bests = {}

    for z in np.linspace(MIN_Z, MAX_Z, z_samples):
        best_z_loss = 0
        for random_sample in range(random_samples):
            params = generate_params({"z": z})
            set_all_params(params)
            image = RENDERER.render()
            with torch.no_grad():
                out = MODEL(image)

            loss = CRITERION(out, LABELS).item()
            if loss < best_loss:
                best_loss = loss
                best_params = params

            if loss < best_z_loss:
                best_z_loss = loss

        z_bests[z] = best_z_loss

    top_2_zs = sorted(z_bests.items(), key=lambda kv: kv[1], reverse=True)[:2]
    just_zs = (top_2_zs[0][0], top_2_zs[0][1])
    best_zs = {"min_z": min(just_zs), "max_z": max(just_zs)}

    return (best_params, best_loss, best_zs)


def generate_params(initial_params={}):
    """Randomly generate translation and rotation parameters.

    :param initial_params:
    :return:
    """
    z = initial_params.get("z", np.random.uniform(MIN_Z, MAX_Z))
    params = {"z": z}
    max_trans = TAN_ANGLE * np.abs(CAMERA_DISTANCE - z) / 2 #  !!! BEFORE WAS max_trans = TAN_ANGLE * np.abs(CAMERA_DISTANCE - z)
    for trans_param in TRANS_PARAMS:
        if trans_param == "z":
            continue
        else:
            params[trans_param] = initial_params.get(
                trans_param, np.random.uniform(-max_trans, max_trans)
            )


    for rot in ROTS:
        for rot_axis in ROT_AXES:
            rot_param = "{0}_{1}".format(rot_axis, rot)
            angle = initial_params.get(rot_param, np.random.uniform(-np.pi, np.pi))
            params[rot_param] = angle
            params[rot_param + "_x"] = np.cos(angle)
            params[rot_param + "_y"] = np.sin(angle)

    return params


def set_all_params(params):
    """Set all renderer parameters.

    :param params:
    :return:
    """
    for trans_param in TRANS_PARAMS:
        RENDERER.prog[trans_param].value = params[trans_param]

    for rot in ROTS:
        angles = {}
        for rot_axis in ROT_AXES:
            rot_param = "{0}_{1}".format(rot_axis, rot)
            angle_x = params[rot_param + "_x"]
            angle_y = params[rot_param + "_y"]
            angles[rot_axis] = np.arctan2(angle_y, angle_x)

        R = gen_rotation_matrix(**angles)
        RENDERER.prog["R_" + rot].write(R.T.astype("f4").tobytes())


def set_param(param, val, params):
    """Set renderer parameter.

    :param param:
    :param val:
    :param params:
    :return:
    """
    if param in TRANS_PARAMS:
        RENDERER.prog[param].value = val
    elif param.split("_")[0] in ROT_AXES:
        [rot_axis, rot, coord] = param.split("_")
        rot_param = "{0}_{1}".format(rot_axis, rot)
        angles = {
            "yaw": params[rot_param],
            "pitch": params[rot_param],
            "roll": params[rot_param],
        }
        if coord == "x":
            angle_x = val
            angle_y = params[rot_param + "_y"]
        else:
            angle_x = params[rot_param + "_x"]
            angle_y = val

        angles[rot_axis] = np.arctan2(angle_y, angle_x)
        R = gen_rotation_matrix(**angles)
        RENDERER.prog["R_" + rot].write(R.T.astype("f4").tobytes())


def approx_partial(param, current_val, params):
    """Compute the approximate partial derivative using the finite-difference method.

    :param param:
    :param current_val:
    :param params:
    :return:
    """
    step_size = STEP_SIZES[param]
    losses = []

    for sign in [-1, 1]:
        set_param(param, current_val + sign * step_size / 2, params)
        image = RENDERER.render()
        with torch.no_grad():
            out = MODEL(image)

        loss = CRITERION(out, LABELS).item()
        losses.append(loss)

    grad = (losses[1] - losses[0]) / step_size
    return grad


def evaluate_params(params):
    """Evaluate renderer parameters.

    :param params:
    :return:
    """
    set_all_params(params)

    image = RENDERER.render()
    with torch.no_grad():
        out = MODEL(image)

    probs = torch.nn.functional.softmax(out, dim=1)
    probs_np = probs[0].detach().cpu().numpy()
    target_prob = probs_np[TARGET_CLASS]

    max_prob = probs_np.max()
    max_index = probs_np.argmax()

    return (image, target_prob, max_prob, max_index)


def run_finite_diff(params, gif_f=None, iterations=100):
    """Run finite-difference optimization.

    :param initial_params:
    :param gif_f:
    :param iterations:
    :return:
    """
    if gif_f is not None:
        writer = imageio.get_writer(gif_f, mode="I")

    start_prob = 0
    best_iter = 0
    best_prob = 0
    best_params = None
    first_hit = -1

    for current_iter in range(iterations):
        logging.info(current_iter)

        (image, target_prob, max_prob, max_index) = evaluate_params(params)
        if current_iter == 0:
            start_prob = target_prob

        if target_prob > best_prob:
            best_iter = current_iter
            best_prob = target_prob
            best_params = params.copy()

        logging.info(target_prob)
        logging.info(start_prob)
        logging.info(best_prob)

        if first_hit == -1 and max_index == TARGET_CLASS:
            first_hit = current_iter

        if gif_f is not None:
            draw = ImageDraw.Draw(image)
            max_label = LABEL_MAP[max_index]
            (w, h) = draw.getfont().getsize(max_label)
            draw.rectangle((0, 0, image.size[0], h), fill="white")
            draw.text(
                (0, 0),
                "{0:.2f}/{1:.2f} - {2}".format(target_prob, max_prob, max_label),
                (0, 0, 0),
            )
            writer.append_data(np.array(image))

        # Calculate approximate partial derivatives.
        grads = {
            param: approx_partial(param, params[param], params)
            for param in UPDATE_PARAMS
        }
        bump = BUMP * np.random.uniform(-1, 1)

        # Update z first.
        if "z" in grads:
            (param, val) = ("z", params["z"])
            val -= LRS[param] * grads[param]
            val = np.clip(val, MIN_Z + bump, MAX_Z + bump)
            params[param] = val

        # Update remaining renderer parameters.
        for param in grads:
            if param == "z":
                continue

            val = params[param]
            val -= LRS[param] * grads[param]
            if param in TRANS_PARAMS:
                max_trans = TAN_ANGLE * np.abs(CAMERA_DISTANCE - params["z"])
                val = np.clip(val, -max_trans + bump, max_trans + bump)

            params[param] = val

        # Normalize angle (x, y)s.
        for rot in ROTS:
            for rot_axis in ROT_AXES:
                rot_param = "{0}_{1}".format(rot_axis, rot)
                angle_x = params[rot_param + "_x"]
                angle_y = params[rot_param + "_y"]
                norm = np.sqrt(angle_x ** 2 + angle_y ** 2)
                params[rot_param + "_x"] = angle_x / norm
                params[rot_param + "_y"] = angle_y / norm
                params[rot_param] = np.arctan2(angle_y, angle_x)

    if gif_f is not None:
        writer.close()

    return (best_prob, best_iter, best_params, first_hit)


def run_z_random_search(params, min_z, max_z, iterations=100):
    """Random sampling within a z range.

    :return:
    """
    start_prob = 0
    best_iter = 0
    best_prob = 0
    best_params = None
    first_hit = -1

    print("TARGET_CLASS: " + str(TARGET_CLASS))
    # TODO - CHANGE THE LOGIC HERE
    while first_hit == -1:
        (image, target_prob, max_prob, max_index) = evaluate_params(params)

        logging.info(target_prob)
        logging.info(start_prob)
        logging.info(best_prob)
        if first_hit == -1 and max_index != TARGET_CLASS and max_prob >= 0.9:
            best_prob = target_prob
            best_params = params.copy()
            first_hit = 1
            break

        z = np.random.uniform(min_z, max_z)
        params = generate_params({"z": z})

    return (best_prob, best_iter, best_params, first_hit)


def dict2array(param_dict):
    """Convert a dictionary of parameters to an array of parameters.

    :param param_dict:
    :return:
    """
    param_array = []
    for param in [
        "x",
        "y",
        "z",
        "yaw_obj_x",
        "yaw_obj_y",
        "pitch_obj_x",
        "pitch_obj_y",
        "roll_obj_x",
        "roll_obj_y",
    ]:
        param_array.append(param_dict[param])

    return np.array(param_array)


def array2dict(param_array):
    """Convert an array of parameters to a dictionary of parameters.

    :param param_array:
    :return:
    """
    param_dict = {
        "x": param_array[0],
        "y": param_array[1],
        "z": param_array[2],
        "yaw_obj_x": param_array[3],
        "yaw_obj_y": param_array[4],
        "pitch_obj_x": param_array[5],
        "pitch_obj_y": param_array[6],
        "roll_obj_x": param_array[7],
        "roll_obj_y": param_array[8],
    }
    return param_dict


def run_cma_es(params, iterations=100):
    """Run CMA-ES optimization.

    :param params:
    :param iterations:
    :return:
    """
    start_prob = 0
    best_iter = 0
    best_prob = 0
    best_params = None
    first_hit = -1
    param_array = dict2array(params)
    es = cma.CMAEvolutionStrategy(param_array, 1.0, {"popsize": 18})
    for current_iter in range(iterations):
        logging.info(current_iter)
        population = es.ask()
        pop_params = [array2dict(individual) for individual in population]
        pop_fitnesses = []
        pop_probs = []
        pop_labels = []
        for params in pop_params:
            set_all_params(params)
            image = RENDERER.render()
            with torch.no_grad():
                out = MODEL(image)

            loss = CRITERION(out, LABELS).item()
            pop_fitnesses.append(loss)

            probs = torch.nn.functional.softmax(out, dim=1)
            probs_np = probs[0].detach().cpu().numpy()
            target_prob = probs_np[TARGET_CLASS]
            pop_probs.append(target_prob)
            max_index = probs_np.argmax()
            pop_labels.append(max_index)

        es.tell(population, pop_fitnesses)

        max_individual = np.argmax(pop_probs)
        target_prob = pop_probs[max_individual]
        if current_iter == 0:
            start_prob = target_prob

        if target_prob > best_prob:
            best_iter = current_iter
            best_prob = target_prob
            best_params = pop_params[max_individual]

        logging.info(target_prob)
        logging.info(start_prob)
        logging.info(best_prob)

        max_index = pop_labels[max_individual]
        if first_hit == -1 and max_index == TARGET_CLASS:
            first_hit = current_iter

    return (best_prob, best_iter, best_params, first_hit)


CLASSES = {
    "jeep": 609,
    "tank": 847,
    "car":817,
    "plane":895,
    "f1car":751,
    "bus":779,
    "train":466,
    "heli":895,
    "boat":403,
    "ufo":812
}

MODELS_PATH = os.path.join(".", "3dmodels")
IMAGES_NUMBER = 100
DATASET_PATH = os.path.join("custom_dataset_v2")

# Images naming convention:
# <classname>_<object number>_i<iteration number>_y<yaw value>_p<pitch value>_r<roll value>.<ext e.x. jpg>
# e.x. bus_1_3_y23_p-13_r_123.jpg

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = Model(device).to(device)
    CRITERION = nn.CrossEntropyLoss()
    RENDERER = Renderer(
        OBJ_PATH, MTL_PATH, camera_distance=CAMERA_DISTANCE, angle_of_view=ANGLE_OF_VIEW
    )
    dir_list = os.listdir(MODELS_PATH)
    dirs = [path for path in dir_list if os.path.isdir(os.path.join(MODELS_PATH, path))]
    os.makedirs(DATASET_PATH, exist_ok=True)
    print(TOO_FAR)
    print(dirs)
    start_time = datetime.now()
    for dir in dirs:
        obj_path = ""
        mtl_path = ""
        for file in os.listdir(os.path.join(MODELS_PATH, dir)):
            if file.endswith('.obj'):
                obj_path = os.path.join(MODELS_PATH, dir, file)
                print(f"Found: {os.path.join(MODELS_PATH, dir, file)}")
            if file.endswith('.mtl'):
                mtl_path = os.path.join(MODELS_PATH, dir, file)
                print(f"Found: {os.path.join(MODELS_PATH, dir, file)}")
        RENDERER.set_up_obj(obj_path, mtl_path)
        print(dir)
        class_name = dir.split("_")[0]
        label_num = CLASSES[class_name]
        TARGET_CLASS = label_num
        print(f"Generating images for: {class_name} (class nr={label_num})")
        LABELS = torch.LongTensor([label_num]).to(device)
        LABEL_MAP = load_imagenet_label_map()
        OPTIM="rs"

        # v2
        if dir.split("_")[-1] == "1":
            IMAGES_NUMBER = 10
        else:
            IMAGES_NUMBER = 40

        for i in range(IMAGES_NUMBER):
            if OPTIM == "fd":
                if INITIAL_PARAMS:
                    start_params = generate_params(INITIAL_PARAMS)
                else:
                    (start_params, start_loss, best_zs) = get_start_params()

                (best_prob, best_iter, best_params, first_hit) = run_finite_diff(
                    start_params.copy(), GIF_F
                )
            elif OPTIM == "zrs":
                (start_params, start_loss, best_zs) = get_start_params()
                (best_prob, best_iter, best_params, first_hit) = run_z_random_search(
                    start_params, best_zs["min_z"], best_zs["max_z"]
                )
            elif OPTIM == "rs":
                    start_params = generate_params()
                    (best_prob, best_iter, best_params, first_hit) = run_z_random_search(
                        start_params, MIN_Z, MAX_Z
                    )
            elif OPTIM == "cma_es":
                (start_params, start_loss, best_zs) = get_start_params()
                (best_prob, best_iter, best_params, first_hit) = run_cma_es(start_params)
            else:
                (start_params, start_loss, best_zs) = get_start_params()
                (best_prob, best_iter, best_params, first_hit) = run_cma_es(start_params)

            print(str(i) + str(best_params))
            # Alter renderer parameters.
            R_obj = gen_rotation_matrix(best_params["yaw_obj"], best_params["pitch_obj"], best_params["roll_obj"])
            RENDERER.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
            RENDERER.prog["x"].value = best_params["x"]
            RENDERER.prog["y"].value = best_params["y"]
            RENDERER.prog["z"].value = best_params["z"]
            RENDERER.prog["amb_int"].value = 0.3
            RENDERER.prog["dif_int"].value = 0.9
            DirLight = np.array([1.0, 1.0, 1.0])
            DirLight /= np.linalg.norm(DirLight)
            RENDERER.prog["DirLight"].value = tuple(DirLight)

            # Render new scene.
            image = RENDERER.render()
            yaw = int(math.degrees(best_params["yaw_obj"]))
            pitch = int(math.degrees(best_params["pitch_obj"]))
            roll = int(math.degrees(best_params["roll_obj"]))
            print(f"Yaw:{yaw}, pitch:{pitch}, roll:{roll}")
            image.save(os.path.join(DATASET_PATH, f"{dir}_i{i}_y{yaw}_p{pitch}_r{roll}.jpg"))
    finish_time = datetime.now()
    print("Start  Time:", start_time.strftime("%H:%M:%S"))
    print("Finish Time:", finish_time.strftime("%H:%M:%S"))


