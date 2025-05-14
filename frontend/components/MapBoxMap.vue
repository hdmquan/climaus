<template>
    <div ref="mapContainer" class="w-full h-full" />
</template>

<script setup>
import { onMounted, ref } from "vue"
import mapboxgl from "mapbox-gl"

const mapContainer = ref()
mapboxgl.accessToken = import.meta.env.MAPBOX_API_KEY

onMounted(() => {
    new mapboxgl.Map({
        container: mapContainer.value,
        style: "mapbox://styles/mapbox/satellite-streets-v11",
        center: [133.7751, -25.2744],
        zoom: 4
    })
})
</script>
