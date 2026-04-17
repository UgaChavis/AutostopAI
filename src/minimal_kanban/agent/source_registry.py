from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDefinition:
    key: str
    label: str
    kind: str
    domains: tuple[str, ...]
    note: str = ""


VIN_SOURCES: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        key="nhtsa_vpic",
        label="NHTSA vPIC",
        kind="vin",
        domains=("vpic.nhtsa.dot.gov",),
        note="Reference VIN source; web research may collect additional supporting evidence.",
    ),
)

PARTS_CATALOG_SOURCES: tuple[SourceDefinition, ...] = ()
PARTS_PRICE_SOURCES: tuple[SourceDefinition, ...] = ()
DTC_SOURCES: tuple[SourceDefinition, ...] = ()
FAULT_SOURCES: tuple[SourceDefinition, ...] = ()


def trusted_domains(*, kind: str) -> list[str]:
    registries = {
        "vin": VIN_SOURCES,
        "catalog": PARTS_CATALOG_SOURCES,
        "price": PARTS_PRICE_SOURCES,
        "dtc": DTC_SOURCES,
        "fault": FAULT_SOURCES,
    }
    return [domain for item in registries.get(kind, ()) for domain in item.domains]


def describe_sources() -> str:
    return "VIN research: NHTSA vPIC (vpic.nhtsa.dot.gov) plus public web evidence collected by the agent"
