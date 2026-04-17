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
        note="Primary VIN decoder.",
    ),
)

PARTS_CATALOG_SOURCES: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        key="partsouq",
        label="PartSouq",
        kind="catalog",
        domains=("partsouq.com",),
        note="OEM catalog and part detail pages.",
    ),
    SourceDefinition(
        key="amayama",
        label="Amayama",
        kind="catalog",
        domains=("amayama.com",),
        note="OEM catalog and cross-reference source.",
    ),
    SourceDefinition(
        key="megazip",
        label="MegaZip",
        kind="catalog",
        domains=("megazip.net",),
        note="OEM catalog source.",
    ),
)

PARTS_PRICE_SOURCES: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        key="emex",
        label="Emex",
        kind="price",
        domains=("emex.ru",),
        note="Market price source.",
    ),
    SourceDefinition(
        key="autopiter",
        label="Autopiter",
        kind="price",
        domains=("autopiter.ru",),
        note="Market price source.",
    ),
    SourceDefinition(
        key="exist",
        label="Exist",
        kind="price",
        domains=("exist.ru",),
        note="Market price source.",
    ),
    SourceDefinition(
        key="zzap",
        label="ZZAP",
        kind="price",
        domains=("zzap.ru",),
        note="Market price source.",
    ),
)

DIAGNOSTIC_SOURCES: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        key="obd_codes",
        label="OBD-Codes",
        kind="dtc",
        domains=("obd-codes.com",),
        note="Trouble code reference.",
    ),
    SourceDefinition(
        key="dtcdecode",
        label="DTCDecode",
        kind="dtc",
        domains=("dtcdecode.com",),
        note="Trouble code lookup reference.",
    ),
    SourceDefinition(
        key="carcarekiosk",
        label="CarCareKiosk",
        kind="fault",
        domains=("carcarekiosk.com",),
        note="Symptom and repair info.",
    ),
)

GENERIC_WEB_SOURCES: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        key="duckduckgo_html",
        label="DuckDuckGo HTML",
        kind="search",
        domains=("duckduckgo.com", "html.duckduckgo.com"),
        note="External search index for discovery.",
    ),
)


def trusted_domains(*, kind: str) -> list[str]:
    registries = {
        "vin": VIN_SOURCES,
        "catalog": PARTS_CATALOG_SOURCES,
        "price": PARTS_PRICE_SOURCES,
        "dtc": DIAGNOSTIC_SOURCES,
        "fault": DIAGNOSTIC_SOURCES,
        "search": GENERIC_WEB_SOURCES,
    }
    return [domain for item in registries.get(kind, ()) for domain in item.domains]


def describe_sources() -> str:
    lines: list[str] = []
    for group_name, items in (
        ("VIN", VIN_SOURCES),
        ("CATALOG", PARTS_CATALOG_SOURCES),
        ("PRICE", PARTS_PRICE_SOURCES),
    ):
        labels = ", ".join(f"{item.label} ({'/'.join(item.domains)})" for item in items)
        lines.append(f"{group_name}: {labels}")
    return "\n".join(lines)
