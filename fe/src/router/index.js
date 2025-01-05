import Vue from 'vue'
import VueRouter from 'vue-router'

Vue.use(VueRouter)

// 解决导航栏或者底部导航tabBar中的vue-router在3.0版本以上频繁点击菜单报错的问题。
const originalPush = VueRouter.prototype.push
VueRouter.prototype.push = function push(location) {
    return originalPush.call(this, location).catch(err => err)
}

const routes = [
    {
        path: '/front',
        name: '首页',
        component: () => import('../views/Front.vue'),
    },
    {path: '*', name: 'NotFound', meta: {name: '无法访问'}, component: () => import('../views/404.vue')},
]

const router = new VueRouter({
    mode: 'hash',
    base: process.env.BASE_URL,
    routes
})

router.beforeEach((to, from, next) => {
    if (to.path === '/' || to.path === '') {
        next('/front')
    } else {
        next()
    }
})

export default router