// pages/index/index.js
// 后端地址：指向你电脑上正在运行的年龄预测服务（通过公网隧道）。
// 注意：这个链接是临时的，重启隧道后会变，变了就改这里。
const API_URL = 'https://mesh-jeans-sender-santa.trycloudflare.com/predict';

Page({
  data: {
    previewPath: '',
    photoPath: '',
    ready: false,
    age: null,
    note: '',
    error: ''
  },

  // 选择照片
  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      success: (res) => {
        const file = res.tempFiles[0];
        this.setData({
          previewPath: file.tempFilePath,
          photoPath: file.tempFilePath,
          ready: true,
          age: null,
          note: '',
          error: ''
        });
      }
    });
  },

  // 上传并预测
  predict() {
    if (!this.data.photoPath) return;
    const self = this;
    this.setData({ error: '', age: null });

    wx.showLoading({ title: '分析中...' });
    wx.uploadFile({
      url: API_URL,
      filePath: this.data.photoPath,
      name: 'photo',
      success(res) {
        wx.hideLoading();
        try {
          const data = JSON.parse(res.data);
          if (res.statusCode === 200 && data.age !== undefined) {
            self.setData({ age: data.age, note: data.note || '' });
          } else {
            self.setData({ error: data.error || '请求失败' });
          }
        } catch (e) {
          self.setData({ error: '返回数据异常' });
        }
      },
      fail() {
        wx.hideLoading();
        self.setData({ error: '请求失败，请确认电脑端的网页服务和公网隧道正在运行' });
      }
    });
  }
});
