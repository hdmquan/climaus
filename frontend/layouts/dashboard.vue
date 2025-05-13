<template>
    <div class="flex h-screen w-screen overflow-hidden bg-background">
        <Sidebar />
        <div class="flex flex-col flex-1">
            <Topbar />
            <div class="flex-1 overflow-hidden">
                <slot />
            </div>
        </div>
    </div>
</template>

<script setup>
import Sidebar from "@/components/Sidebar.vue"
import Topbar from "@/components/Topbar.vue"
</script>
