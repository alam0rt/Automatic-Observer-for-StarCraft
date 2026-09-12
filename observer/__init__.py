"""Automatic observer for StarCraft: Brood War.

Pipeline:
    extract   replays -> per-game units / events / fights tables (sc-extract, OpenBW + BWEM + FAP)
    features  game state rasterised onto the map grid
    labels    where the action is: hindsight (what actually died) plus FAP's predicted losses
    model     heatmap network trained to predict those labels from past frames only
    camera    predicted heatmaps -> a smooth viewport track (.vpd)
"""
