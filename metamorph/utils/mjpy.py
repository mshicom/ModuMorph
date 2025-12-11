import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from lxml import etree


def _qpos_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _qvel_width(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def _build_name_list(model: mujoco.MjModel, obj_type: int, count: int) -> Tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, obj_type, idx) or ""
        for idx in range(count)
    )


class ModelWrapper:
    """Lightweight shim to mirror mujoco_py's Model helpers."""

    def __init__(self, model: mujoco.MjModel):
        self._model = model
        self.joint_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        self.body_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        self.geom_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
        self.site_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
        self.sensor_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)
        self.camera_names = _build_name_list(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        self._camera_name2id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.camera_names) if name
        }

    # ------------------------------------------------------------------
    # Name lookup helpers
    # ------------------------------------------------------------------
    def site_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)

    def site_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, idx)

    def geom_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def geom_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, idx)

    def body_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

    def body_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, idx)

    def sensor_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    def sensor_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SENSOR, idx)

    def camera_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    # ------------------------------------------------------------------
    # Address helpers
    # ------------------------------------------------------------------
    def get_joint_qpos_addr(self, joint_name: str):
        j_id = self.joint_name2id(joint_name)
        addr = int(self._model.jnt_qposadr[j_id])
        width = _qpos_width(int(self._model.jnt_type[j_id]))
        return (addr, addr + width) if width > 1 else addr

    def get_joint_qvel_addr(self, joint_name: str):
        j_id = self.joint_name2id(joint_name)
        addr = int(self._model.jnt_dofadr[j_id])
        width = _qvel_width(int(self._model.jnt_type[j_id]))
        return (addr, addr + width) if width > 1 else addr

    def joint_name2id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def joint_id2name(self, idx: int) -> str:
        return mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, idx)

    def __getattr__(self, item):
        return getattr(self._model, item)


class DataWrapper:
    """Expose mujoco.MjData attributes with mujoco_py-like helpers."""

    def __init__(self, model: ModelWrapper, data: mujoco.MjData):
        self._model = model
        self._data = data

    def __getattr__(self, item):
        return getattr(self._data, item)

    # Convenience properties mirroring mujoco_py
    @property
    def body_xpos(self):
        return self._data.xpos

    @property
    def body_xmat(self):
        return self._data.xmat

    @property
    def body_xquat(self):
        return self._data.xquat

    @property
    def body_xvelp(self):
        # Linear velocity in global frame
        return self._data.cvel[:, 3:]

    @property
    def body_xvelr(self):
        # Angular velocity in global frame
        return self._data.cvel[:, :3]

    @property
    def geom_xpos(self):
        return self._data.geom_xpos

    @property
    def site_xpos(self):
        return self._data.site_xpos

    def get_body_xpos(self, name: str):
        idx = self._model.body_name2id(name)
        return self._data.xpos[idx].copy()

    def get_body_xmat(self, name: str):
        idx = self._model.body_name2id(name)
        return self._data.xmat[idx].copy()

    def get_site_xpos(self, name: str):
        idx = self._model.site_name2id(name)
        return self._data.site_xpos[idx].copy()


@dataclass
class MjSimState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    udd_state: Optional[Dict] = None


class MjSim:
    """Minimal simulation wrapper compatible with previous mujoco_py usage."""

    def __init__(self, model: mujoco.MjModel):
        self.model = ModelWrapper(model)
        self.data = DataWrapper(self.model, mujoco.MjData(model))
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self.forward()

    # Simulation lifecycle -------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model._model, self.data._data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.forward()

    def step(self):
        mujoco.mj_step(self.model._model, self.data._data)

    def forward(self):
        mujoco.mj_forward(self.model._model, self.data._data)

    def get_state(self) -> MjSimState:
        return MjSimState(
            self.data.time,
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.act.copy(),
            udd_state=None,
        )

    def set_state(self, state: MjSimState):
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if state.act is not None:
            self.data.act[:] = state.act
        self.forward()


def _select_jax_device(device_name: Optional[str]):
    if device_name is None or device_name == "":
        return None
    normalized = device_name.lower()
    for dev in jax.devices():
        full_name = f"{dev.platform}:{dev.id}"
        if normalized == full_name or normalized == dev.platform or normalized == str(dev.id):
            return dev
        if normalized == dev.device_kind.lower():
            return dev
    raise ValueError(f"Requested JAX device '{device_name}' not found among {[f'{d.platform}:{d.id}' for d in jax.devices()]}")


class MjxSim:
    """Simulation wrapper backed by mujoco.mjx for GPU-accelerated stepping."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        device: Optional[str] = None,
        impl: Optional[str] = None,
        use_jit: bool = True,
    ):
        self.model = ModelWrapper(model)
        self.data = DataWrapper(self.model, mujoco.MjData(model))
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._device = _select_jax_device(device)
        self._impl = impl
        self._mjx_model = mjx.put_model(model, device=self._device, impl=impl)
        self._mjx_data = mjx.make_data(self._mjx_model)
        self._refresh_mjx_data_from_host()
        self._step_fn = jax.jit(self._step_impl) if use_jit else self._step_impl
        self._forward_fn = jax.jit(self._forward_impl) if use_jit else self._forward_impl
        self.forward()

    def _device_put(self, arr):
        return jax.device_put(arr, self._device) if self._device else jnp.asarray(arr)

    def _refresh_mjx_data_from_host(self):
        self._mjx_data = self._mjx_data.replace(
            qpos=self._device_put(self.data.qpos),
            qvel=self._device_put(self.data.qvel),
            act=self._device_put(self.data.act),
            time=self.data.time,
        )

    def _sync_from_mjx(self):
        self.data._data = mjx.get_data(self.model._model, self._mjx_data)

    def _step_impl(self, mjx_data: mjx.Data, ctrl: jnp.ndarray):
        return mjx.step(self._mjx_model, mjx_data.replace(ctrl=ctrl))

    def _forward_impl(self, mjx_data: mjx.Data):
        return mjx.forward(self._mjx_model, mjx_data)

    def reset(self):
        mujoco.mj_resetData(self.model._model, self.data._data)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self._mjx_data = mjx.make_data(self._mjx_model)
        self._refresh_mjx_data_from_host()
        self.forward()

    def step(self):
        # ctrl is updated in-place by callers, so convert on every step before dispatching to JAX
        ctrl = self._device_put(self.data.ctrl)
        self._mjx_data = self._step_fn(self._mjx_data, ctrl)
        self._sync_from_mjx()

    def forward(self):
        self._mjx_data = self._forward_fn(self._mjx_data)
        self._sync_from_mjx()

    def get_state(self) -> MjSimState:
        return MjSimState(
            self.data.time,
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.act.copy(),
            udd_state=None,
        )

    def set_state(self, state: MjSimState):
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        if state.act is not None:
            self.data.act[:] = state.act
        self._refresh_mjx_data_from_host()
        self.forward()


def make_sim(
    model: mujoco.MjModel,
    backend: str = "mujoco",
    *,
    mjx_device: Optional[str] = None,
    mjx_impl: Optional[str] = None,
    mjx_jit: bool = True,
):
    normalized = backend.lower()
    impl = None if mjx_impl == "" else mjx_impl
    if normalized == "mujoco":
        return MjSim(model)
    if normalized == "mjx":
        return MjxSim(model, device=mjx_device, impl=impl, use_jit=mjx_jit)
    raise ValueError(f"Unsupported Mujoco backend '{backend}'")


class OffscreenViewer:
    """Offscreen renderer built on the new mujoco API."""

    def __init__(self, sim: MjSim, device_id: int = -1):
        del device_id  # Kept for API compatibility
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model._model)
        self._last_rgb = None
        self._last_depth = None
        self._markers = []

    def render(self, width: int, height: int, camera_id: Optional[int] = None):
        self.renderer.update_scene(self.sim.data._data, camera_id=camera_id)
        rgb, depth = self.renderer.render(width=width, height=height, depth=True)
        self._last_rgb = rgb
        self._last_depth = depth

    def read_pixels(self, width: int, height: int, depth: bool = False):
        # width/height kept for API parity
        del width, height
        if depth:
            return self._last_rgb, self._last_depth
        return self._last_rgb

    def add_marker(self, **kwargs):
        self._markers.append(kwargs)

    def update_sim(self, sim: MjSim):
        self.sim = sim
        self.renderer = mujoco.Renderer(sim.model._model)


class HumanViewer:
    """Simple wrapper around mujoco.viewer for human rendering."""

    def __init__(self, sim: MjSim):
        self.sim = sim
        self._viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)
        self._markers = []

    def render(self, *_, **__):
        if self._viewer is None:
            return
        if hasattr(self._viewer, "is_running") and not self._viewer.is_running():
            return
        self._viewer.sync()

    def add_marker(self, **kwargs):
        # The new viewer does not expose add_marker; store for parity.
        self._markers.append(kwargs)

    def update_sim(self, sim: MjSim):
        self.sim = sim
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = mujoco.viewer.launch_passive(sim.model._model, sim.data._data)


def mj_name2id(sim, type_, name):
    """Returns the mujoco id corresponding to name."""
    if type_ == "site":
        return sim.model.site_name2id(name)
    elif type_ == "geom":
        return sim.model.geom_name2id(name)
    elif type_ == "body":
        return sim.model.body_name2id(name)
    elif type_ == "sensor":
        return sim.model.sensor_name2id(name)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mj_id2name(sim, type_, id_):
    """Returns the mujoco name corresponding to id."""
    if type_ == "site":
        return sim.model.site_id2name(id_)
    elif type_ == "geom":
        return sim.model.geom_id2name(id_)
    elif type_ == "body":
        return sim.model.body_id2name(id_)
    elif type_ == "sensor":
        return sim.model.sensor_id2name(id_)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mjsim_from_etree(root):
    """Return MjSim from etree root."""
    return MjSim(mjmodel_from_etree(root))


def mjmodel_from_etree(root):
    """Return MjModel from etree root."""
    model_string = etree.tostring(root, encoding="unicode", pretty_print=True)
    return load_model_from_xml(model_string)


def load_model_from_xml(xml_string: str) -> mujoco.MjModel:
    """Load an MjModel from an XML string using the new mujoco API."""
    return mujoco.MjModel.from_xml_string(xml_string)


def joint_qpos_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qpos values."""
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qpos_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qpos values of all joints matching the prefix."""
    qpos_idxs_list = [
        joint_qpos_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def qpos_idxs_for_agent(sim):
    """Gets indexes for the qpos values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qpos_idxs_list = [joint_qpos_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def joint_qvel_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qvel values."""
    addr = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qvel_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qvel values of all joints matching the prefix."""
    qvel_idxs_list = [
        joint_qvel_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def qvel_idxs_for_agent(sim):
    """Gets indexes for the qvel values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qvel_idxs_list = [joint_qvel_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def geom_idxs_for_agent(sim):
    """Gets indexes for agent geoms."""
    agent_geoms = names_from_prefixes(sim, ["torso", "limb"], "geom")
    geom_idx_list = [
        mj_name2id(sim, "geom", geom_name) for geom_name in agent_geoms
    ]
    return geom_idx_list


def body_idxs_for_agent(sim):
    """Gets indexes for agent body."""
    agent_bodies = names_from_prefixes(sim, ["torso", "limb"], "body")
    body_idx_list = [
        mj_name2id(sim, "body", body_name) for body_name in agent_bodies
    ]
    return body_idx_list


def names_from_prefixes(sim, prefixes, elem_type):
    """Get all names of elem_type elems which match any of the prefixes."""
    all_names = getattr(sim.model, "{}_names".format(elem_type))
    matches = []
    for name in all_names:
        for prefix in prefixes:
            if name.startswith(prefix):
                matches.append(name)
                break
    return matches


def get_active_contacts(sim):
    num_contacts = sim.data.ncon
    contacts = sim.data.contact[:num_contacts]
    contact_geoms = [
        tuple(
            sorted(
                (
                    mj_id2name(sim, "geom", contact.geom1),
                    mj_id2name(sim, "geom", contact.geom2),
                )
            )
        )
        for contact in contacts
    ]
    return sorted(list(set(contact_geoms)))
