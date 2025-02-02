"""Tuya Energy Meter."""

from collections.abc import Callable
from typing import Any, Final, Optional

from zigpy.quirks.v2.homeassistant import EntityType, PERCENTAGE, UnitOfTime
import zigpy.types as t
from zigpy.zcl import Cluster
from zigpy.zcl.clusters.homeautomation import MeasurementType
from zigpy.zcl.foundation import BaseAttributeDefs, ZCLAttributeDef

from zhaquirks import LocalDataCluster
from zhaquirks.tuya import (
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
    TuyaZBMeteringClusterWithUnit,
)
from zhaquirks.tuya.builder import TuyaQuirkBuilder
from zhaquirks.tuya.mcu import TuyaMCUCluster


POWER_FLOW: Final = "power_flow"


class Channel(t.enum8):
    """Enum for meter channel endpoint_id."""

    A = 1
    B = 2
    C = 3
    AB = 11

    @classmethod
    def attr_suffix(cls, channel: Optional["Channel"]) -> str:
        """Return the attribute suffix for a channel."""
        return cls.__CHANNEL_ATTR_SUFFIX.get(channel, "")

    @classmethod
    @property
    def virtual(cls) -> set["Channel"]:
        """Return set of virtual channels."""
        return cls.__VIRTUAL_CHANNELS

    __CHANNEL_ATTR_SUFFIX: dict["Channel", str] = {
        B: "_ch_b",
        C: "_ch_c",
    }
    __VIRTUAL_CHANNELS: set["Channel"] = {AB}


class PowerFlow(t.enum1):
    """Enum for power flow direction."""

    Forward = 0x0
    Reverse = 0x1

    @classmethod
    def align_value(
        cls, value: int | float, power_flow: Optional["PowerFlow"] = None
    ) -> int:
        """Align the value with power_flow direction."""
        if (
            power_flow == cls.Reverse
            and value > 0
            or power_flow == cls.Forward
            and value < 0
        ):
            value = -value
        return value


class TuyaPowerPhase:
    """Extract values from a Tuya power phase datapoint."""

    @staticmethod
    def variant_1(value) -> tuple[t.uint_t, t.uint_t]:
        """Variant 1 of power phase Data Point."""
        voltage = value[14] | value[13] << 8
        current = value[12] | value[11] << 8
        return voltage, current

    @staticmethod
    def variant_2(value) -> tuple[t.uint_t, t.uint_t, int]:
        """Variant 2 of power phase Data Point."""
        voltage = value[1] | value[0] << 8
        current = value[4] | value[3] << 8
        power = value[7] | value[6] << 8
        return voltage, current, power * 10

    @staticmethod
    def variant_3(value) -> tuple[t.uint_t, t.uint_t, int]:
        """Variant 3 of power phase Data Point."""
        voltage = (value[0] << 8) | value[1]
        current = (value[2] << 16) | (value[3] << 8) | value[4]
        power = (value[5] << 16) | (value[6] << 8) | value[7]
        return voltage, current, power * 10


class ChannelConfiguration(t.enum8):
    """Enum for all energy meter channel configurations."""

    none = 0
    A_plus_B = 1
    A_minus_B = 2
    B_minus_A = 3


class EnergyMeterConfig(LocalDataCluster):
    """Local cluster for storing meter configuration."""

    cluster_id: Final[t.uint16_t] = 0xFC00
    name: Final = "Energy Meter Config"
    ep_attribute: Final = "energy_meter_config"

    ChannelConfiguration: Final = ChannelConfiguration

    class AttributeDefs(BaseAttributeDefs):
        """Manufacturer specific attributes."""

        channel_configuration = ZCLAttributeDef(
            id=0x5000,
            type=ChannelConfiguration,
            access="rw",
            is_manufacturer_specific=True,
        )
        power_flow_mitigation = ZCLAttributeDef(
            id=0x5010,
            type=t.Bool,
            access="rw",
            is_manufacturer_specific=True,
        )


class ChannelHelper:
    """Common methods for energy meter clusters."""

    _EXTENSIVE_ATTRIBUTES: tuple[str] = ()

    @property
    def channel(self) -> Channel | None:
        """Return the cluster channel."""
        try:
            return Channel(self.endpoint.endpoint_id)
        except ValueError:
            return None

    def clear_attributes(self):
        """Clear the cached value of cluster attributes."""
        for attr in self.AttributeDefs:
            super().update_attribute(attr.name, None)

    def get_cluster(
        self,
        endpoint_id: int,
        ep_attribute: str | None = None,
    ) -> Cluster:
        """Return the cluster for the given endpoint, default to current cluster type."""
        return getattr(
            self.endpoint.device.endpoints[endpoint_id],
            ep_attribute or self.ep_attribute,
        )

    @property
    def mcu_cluster(self) -> TuyaMCUCluster | None:
        """Return the MCU cluster."""
        return getattr(
            self.endpoint.device.endpoints[1], TuyaMCUCluster.ep_attribute, None
        )

    @property
    def meter_config(self, attr_name: str, default: Any = None) -> Any:
        """Return the config attribute value."""
        cluster = getattr(
            self.endpoint.device.endpoints[1], EnergyMeterConfig.ep_attribute, None
        )
        if not cluster:
            return default
        return cluster.get(attr_name, default)

class PowerFlowHelper(ChannelHelper):
    """Apply Tuya power_flow to ZCL power attributes."""

    UNSIGNED_ATTR_SUFFIX: Final = "_attr_unsigned"

    @property
    def power_flow(self) -> PowerFlow | None:
        """Return the channel power flow direction."""
        if not self.mcu_cluster:
            return None
        try:
            return self.mcu_cluster.get(POWER_FLOW + Channel.attr_suffix(self.channel))
        except KeyError:
            return None

    @power_flow.setter
    def power_flow(self, value: PowerFlow) -> None:
        """Update the channel power flow direction."""
        if not self.mcu_cluster:
            return None
        self.mcu_cluster.update_attribute(
            POWER_FLOW + Channel.attr_suffix(self.channel)
        )

    def power_flow_handler(self, attr_name: str, value) -> tuple[str, Any]:
        """Orchestrate processing of directional attributes."""
        attr_name, value = self._align_unsigned_attribute(attr_name, value)
        return attr_name, value

    def _align_unsigned_attribute(self, attr_name: str, value) -> tuple[str, Any]:
        """Unsigned attributes are aligned with power flow direction."""
        if attr_name.endswith(self.UNSIGNED_ATTR_SUFFIX):
            attr_name = attr_name.removesuffix(self.UNSIGNED_ATTR_SUFFIX)
            value = PowerFlow.align_value(value, self.power_flow)
        return attr_name, value


class PowerFlowMitigation(PowerFlowHelper, ChannelHelper):
    """Logic compensating for delayed power flow direction reporting.

    _TZE204_81yrt3lo (app_version: 74, hw_version: 1 and stack_version: 0) has a bug
    which results in it reporting power_flow after its power data points.
    This means a change in direction would only be reported after the subsequent DP report,
    resulting in incorrect attribute signing in the ZCL clusters.
    
    This mitigation holds attribute update values until the subsequent power_flow report,
    resulting in correct values, but a delay in attribute update equal to the update interval.
    """

    HOLD = "hold"
    RELEASE = "release"

    """Devices requiring power flow mitigation."""
    _POWER_FLOW_MITIGATION: tuple[dict] = (
        {
            "manufacturer": "_TZE204_81yrt3lo",
            "model": "TS0601",
            "basic_cluster": {
                "app_version": 74,
                "hw_version": 1,
                "stack_version": 0,
            },
        },
    )

    def __init__(self, *args, **kwargs):
        """Init."""
        self._held_values: dict[str, Any] = {}
        self._mitigation_required: bool | None = None
        super().__init__(*args, **kwargs)

    def power_flow_mitigation_handler(self, attr_name: str, value) -> str | None:
        """Compensate for delay in reported power flow direction."""

        if not self.power_flow_mitigation_required or not self.power_flow_mitigation:
            return None

    @property
    def power_flow_mitigation(self) -> bool:
        """Return True if the mitigation setting is enabled."""
        return self.meter_config(
            EnergyMeterConfig.AttributeDefs.power_flow_mitigation.name, False
        )

    @property
    def power_flow_mitigation_required(self) -> bool:
        """Return True if the device requires Power Flow mitigations."""
        if self._mitigation_required is None:
            self._mitigation_required = self._evaluate_device_mitigation()
        return self._mitigation_required

    def _mitigation_action(
        self, attr_name: str, value: int, trigger_channel: Channel
    ) -> str:
        """Return the action for the power flow mitigation handler."""
        return self.RELEASE

    def _get_held_value(self, attr_name: str) -> int | None:
        """Retrieve the held attribute value."""
        return self._held_values.get(attr_name, None)

    def _store_value(self, attr_name: str, value: int | None) -> None:
        """Store the update value."""
        self._held_values[attr_name] = value

    def _release_held_values(
        self, attr_name: str, source_channels: tuple[Channel], trigger_channel: Channel
    ) -> None:
        """Release held values to update the cluster attributes."""
        for channel in source_channels:
            cluster = self.get_cluster(channel)
            if channel != trigger_channel:
                value = cluster._get_held_value(attr_name)
                if value is not None:
                    cluster.update_attribute(attr_name, value)
            cluster._store_value(attr_name, None)

    def _evaluate_device_mitigation(self) -> bool:
        """True if the device requires Power Flow mitigation."""
        basic_cluster = self.endpoint.device.endpoints[1].basic
        return {
            "manufacturer": self.endpoint.device.manufacturer,
            "model": self.endpoint.device.model,
            "basic_cluster": {
                "app_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.app_version.name
                ),
                "hw_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.hw_version.name
                ),
                "stack_version": basic_cluster.get(
                    basic_cluster.AttributeDefs.stack_version.name
                ),
            },
        } in self._POWER_FLOW_MITIGATION


class VirtualChannelHelper(ChannelHelper):
    """Methods for calculating virtual energy meter channel attributes."""

    @property
    def channel_configuration(self) -> ChannelConfiguration | None:
        """Return the channel configuration."""
        return self.meter_config(
            EnergyMeterConfig.AttributeDefs.channel_configuration.name, None
        )

    def virtual_channel_handler(self, attr_name: str) -> None:
        """Handle updates to virtual energy meter channels."""

        if attr_name not in self._EXTENSIVE_ATTRIBUTES:
            return None
        for channel in self._virtual_channels:
            trigger_channel, method = self._VIRTUAL_CHANNEL_CONFIGURATION.get(
                (channel, self.channel_configuration), None
            )
            if self.channel != trigger_channel:
                continue
            value = method(attr_name) if method else None
            virtual_cluster = self.get_cluster(channel)
            virtual_cluster.update_attribute(attr_name, value)

    def _is_attr_uint(self, attr_name: str) -> bool:
        """True if the attribute type is an unsigned integer."""
        return issubclass(getattr(self.AttributeDefs, attr_name).type, t.uint_t)

    def _retrieve_source_values(
        self, attr_name: str, channels: tuple[Channel]
    ) -> tuple:
        """Retrieve source values from channels."""
        return tuple(
            PowerFlow.align_value(cluster.get(attr_name), cluster.power_flow)
            if attr_name in self._EXTENSIVE_ATTRIBUTES and self._is_attr_uint(attr_name)
            else cluster.get(attr_name)
            for channel in channels
            for cluster in [self.get_cluster(channel)]
        )

    @property
    def _virtual_channels(self) -> set[Channel]:
        """Virtual channels present on the device."""
        return Channel.virtual.intersection(self.endpoint.device.endpoints.keys())

    def _virtual_a_plus_b(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_plus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_a + value_b

    def _virtual_a_minus_b(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_minus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_a - value_b

    def _virtual_b_minus_a(self, attr_name: str) -> int | None:
        """Calculate virtual channel value for A_minus_B configuration."""
        value_a, value_b = self._retrieve_source_values(
            attr_name, (Channel.A, Channel.B)
        )
        if None in (value_a, value_b):
            return None
        return value_b - value_a

    """Map of virtual channels to their trigger channel and calculation method."""
    _VIRTUAL_CHANNEL_CONFIGURATION: dict[
        tuple[Channel, ChannelConfiguration | None], tuple[Channel, Callable | None]
    ] = {
        (Channel.AB, ChannelConfiguration.A_plus_B): (Channel.B, _virtual_a_plus_b),
        (Channel.AB, ChannelConfiguration.A_minus_B): (Channel.B, _virtual_a_minus_b),
        (Channel.AB, ChannelConfiguration.B_minus_A): (Channel.B, _virtual_b_minus_a),
        (Channel.AB, ChannelConfiguration.none): (Channel.B, None),
    }


class TuyaElectricalMeasurement(
    VirtualChannelHelper,
    PowerFlowMitigation,
    PowerFlowHelper,
    ChannelHelper,
    TuyaLocalCluster,
    TuyaZBElectricalMeasurement,
):
    """ElectricalMeasurement cluster for Tuya energy meter devices."""

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBElectricalMeasurement._CONSTANT_ATTRIBUTES,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_divisor.id: 100,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_frequency_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_power_multiplier.id: 1,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_divisor.id: 10,
        TuyaZBElectricalMeasurement.AttributeDefs.ac_voltage_multiplier.id: 1,
    }

    _ATTRIBUTE_MEASUREMENT_TYPES: dict[str, MeasurementType] = {
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_b.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_c.name: MeasurementType.Active_measurement_AC
        | MeasurementType.Phase_C_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_b.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_c.name: MeasurementType.Reactive_measurement_AC
        | MeasurementType.Phase_C_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_A_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_b.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_B_measurement,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_c.name: MeasurementType.Apparent_measurement_AC
        | MeasurementType.Phase_C_measurement,
    }

    _EXTENSIVE_ATTRIBUTES: tuple[str] = (
        TuyaZBElectricalMeasurement.AttributeDefs.active_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.active_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.apparent_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.reactive_power_ph_c.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current_ph_b.name,
        TuyaZBElectricalMeasurement.AttributeDefs.rms_current_ph_c.name,
    )

    def update_attribute(self, attr_name: str, value) -> None:
        """Update the cluster attribute."""
        if (
            self.power_flow_mitigation_handler(attr_name, value)
            == PowerFlowMitigation.HOLD
        ):
            return None
        attr_name, value = self.power_flow_handler(attr_name, value)

        super().update_attribute(attr_name, value)
        self._update_measurement_type(attr_name)
        self.virtual_channel_handler(attr_name)

    def _update_measurement_type(self, attr_name: str) -> None:
        """Derives the measurement_type from reported attributes."""
        if attr_name not in self._ATTRIBUTE_MEASUREMENT_TYPES:
            return None
        measurement_type = 0
        for measurement, mask in self._ATTRIBUTE_MEASUREMENT_TYPES.items():
            if measurement == attr_name or self.get(measurement) is not None:
                measurement_type |= mask
        super().update_attribute(
            self.AttributeDefs.measurement_type.name, measurement_type
        )


class TuyaMetering(
    VirtualChannelHelper,
    PowerFlowMitigation,
    PowerFlowHelper,
    ChannelHelper,
    TuyaLocalCluster,
    TuyaZBMeteringClusterWithUnit,
):
    """Metering cluster for Tuya energy meter devices."""

    @staticmethod
    def format(
        whole_digits: int, dec_digits: int, suppress_leading_zeros: bool = True
    ) -> int:
        """Return the formatter value for summation and demand Metering attributes."""
        assert 0 <= whole_digits <= 7, "must be within range of 0 to 7."
        assert 0 <= dec_digits <= 7, "must be within range of 0 to 7."
        return (suppress_leading_zeros << 6) | (whole_digits << 3) | dec_digits

    _CONSTANT_ATTRIBUTES: dict[int, Any] = {
        **TuyaZBMeteringClusterWithUnit._CONSTANT_ATTRIBUTES,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.status.id: 0x00,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.multiplier.id: 1,
        TuyaZBMeteringClusterWithUnit.AttributeDefs.divisor.id: 10000,  # 1 decimal place after conversion from kW to W
        TuyaZBMeteringClusterWithUnit.AttributeDefs.summation_formatting.id: format(
            whole_digits=7, dec_digits=2
        ),
        TuyaZBMeteringClusterWithUnit.AttributeDefs.demand_formatting.id: format(
            whole_digits=7, dec_digits=1
        ),
    }

    _EXTENSIVE_ATTRIBUTES: tuple[str] = (
        TuyaZBMeteringClusterWithUnit.AttributeDefs.instantaneous_demand.name,
    )

    def update_attribute(self, attr_name: str, value) -> None:
        """Update the cluster attribute."""
        if (
            self.power_flow_mitigation_handler(attr_name, value)
            == PowerFlowMitigation.HOLD
        ):
            return None
        attr_name, value = self.power_flow_handler(attr_name, value)
        super().update_attribute(attr_name, value)
        self.virtual_channel_handler(attr_name)

(
    ### Tuya PJ-1203A 2 channel bidirectional energy meter
    TuyaQuirkBuilder("_TZE284_81yrt3lo", "TS0601")
    # .tuya_enchantment()
    .adds_endpoint(Channel.A)
    .adds_endpoint(Channel.B)
    .adds_endpoint(Channel.AB)
    .adds(EnergyMeterConfig)
    .adds(TuyaElectricalMeasurement)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.A)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.B)
    .adds(TuyaElectricalMeasurement, endpoint_id=Channel.AB)
    .adds(TuyaMetering)
    .adds(TuyaMetering, endpoint_id=Channel.A)
    .adds(TuyaMetering, endpoint_id=Channel.B)
    .adds(TuyaMetering, endpoint_id=Channel.AB)
    .enum(
        EnergyMeterConfig.AttributeDefs.channel_configuration.name,
        ChannelConfiguration,
        EnergyMeterConfig.cluster_id,
        entity_type=EntityType.CONFIG,
        translation_key="channel_configuration",
        fallback_name="Channel Configuration",
    )
    .tuya_dp_attribute(
        dp_id=102,
        attribute_name=POWER_FLOW + Channel.attr_suffix(Channel.A),
        type=PowerFlow,
        converter=lambda x: PowerFlow(x),
    )
    .tuya_dp_attribute(
        dp_id=104,
        attribute_name=POWER_FLOW + Channel.attr_suffix(Channel.B),
        type=PowerFlow,
        converter=lambda x: PowerFlow(x),
    )
    .tuya_dp(
        dp_id=111,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.ac_frequency.name,
    )
    .tuya_dp(
        dp_id=106,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.A,
    )
    .tuya_dp(
        dp_id=108,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_delivered.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=107,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.A,
    )
    .tuya_dp(
        dp_id=109,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.current_summ_received.name,
        converter=lambda x: x * 100,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=101,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        endpoint_id=Channel.A,
    )
    .tuya_dp(
        dp_id=105,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=115,
        ep_attribute=TuyaMetering.ep_attribute,
        attribute_name=TuyaMetering.AttributeDefs.instantaneous_demand.name
        + PowerFlowHelper.UNSIGNED_ATTR_SUFFIX,
        endpoint_id=Channel.AB,
    )
    .tuya_dp(
        dp_id=110,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.A,
    )
    .tuya_dp(
        dp_id=121,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.power_factor.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=113,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.A,
    )
    .tuya_dp(
        dp_id=114,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_current.name,
        endpoint_id=Channel.B,
    )
    .tuya_dp(
        dp_id=112,
        ep_attribute=TuyaElectricalMeasurement.ep_attribute,
        attribute_name=TuyaElectricalMeasurement.AttributeDefs.rms_voltage.name,
    )
    .tuya_number(
        dp_id=129,
        attribute_name="update_interval",
        type=t.uint32_t_be,
        unit=UnitOfTime.SECONDS,
        min_value=5,
        max_value=60,
        step=1,
        translation_key="update_interval",
        fallback_name="Update Interval",
        entity_type=EntityType.CONFIG,
    )
    .add_to_registry()
)
