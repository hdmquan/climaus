<template>
    <section class="max-w-5xl mx-auto p-8 text-center">
        <h1 class="text-4xl font-bold text-primary mb-4">Welcome to ClimAus</h1>
        <p class="mb-8 text-lg text-gray-600">
            Monitoring environmental change across Australia using satellite intelligence.
        </p>
        <NuxtLink
            to="/dashboard"
            class="bg-primary text-white px-6 py-3 rounded hover:bg-secondary"
        >
            Go to Dashboard
        </NuxtLink>
    </section>
</template>

<script setup>
definePageMeta({ layout: "default", title: "Climaus | Environmental Intelligence" })
</script>
