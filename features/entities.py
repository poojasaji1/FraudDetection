from feast import Entity

card = Entity(
    name="card",
    join_keys=["card1"],
    description="Proxy for unique user — card number prefix",
)
