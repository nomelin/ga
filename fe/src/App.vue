<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>


export default {
  name: 'App'
}


</script>
<style>
:root {
  /* 默认字体大小 */
  --base-font-size: 16px;
}

html {
  /* 使用 CSS 变量设置根元素字体大小 */
  font-size: var(--base-font-size);
}


/*以下的样式是为了解决弹出框导致页面元素发生偏移的问题。*/
body {

  /*这段代码是为了防止页面出现水平滚动条*/
  padding-right: 0 !important;
}

.modal-open {
  overflow-y: scroll;
  padding-right: 0 !important
}


</style>

