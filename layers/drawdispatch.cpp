/* Copyright (c) 2015-2019 The Khronos Group Inc.
 * Copyright (c) 2015-2019 Valve Corporation
 * Copyright (c) 2015-2019 LunarG, Inc.
 * Copyright (C) 2015-2019 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Cody Northrop <cnorthrop@google.com>
 * Author: Michael Lentine <mlentine@google.com>
 * Author: Tobin Ehlis <tobine@google.com>
 * Author: Chia-I Wu <olv@google.com>
 * Author: Chris Forbes <chrisf@ijw.co.nz>
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Ian Elliott <ianelliott@google.com>
 * Author: Dave Houlton <daveh@lunarg.com>
 * Author: Dustin Graves <dustin@lunarg.com>
 * Author: Jeremy Hayes <jeremy@lunarg.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Karl Schultz <karl@lunarg.com>
 * Author: Mark Young <marky@lunarg.com>
 * Author: Mike Schuchardt <mikes@lunarg.com>
 * Author: Mike Weiblen <mikew@lunarg.com>
 * Author: Tony Barbour <tony@LunarG.com>
 * Author: John Zulauf <jzulauf@lunarg.com>
 * Author: Shannon McPherson <shannon@lunarg.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include "chassis.h"
#include "core_validation.h"

// Generic function to handle state update for all CmdDraw* and CmdDispatch* type functions
void CoreChecks::UpdateStateCmdDrawDispatchType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point) {
    UpdateDrawState(cb_state, bind_point);
}

// Generic function to handle state update for all CmdDraw* type functions
void CoreChecks::UpdateStateCmdDrawType(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint bind_point) {
    UpdateStateCmdDrawDispatchType(cb_state, bind_point);
    cb_state->hasDrawCmd = true;
}
void CoreChecks::PostCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawDispatchType(cb_state, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void CoreChecks::PostCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawDispatchType(cb_state, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void CoreChecks::PostCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
    uint32_t firstVertex, uint32_t firstInstance) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PostCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
    uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PostCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
    uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PostCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    uint32_t count, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PreCallRecordCmdDrawIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
    uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PreCallRecordCmdDrawIndexedIndirectCountKHR(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    VkBuffer countBuffer, VkDeviceSize countBufferOffset,
    uint32_t maxDrawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PreCallRecordCmdDrawMeshTasksNV(VkCommandBuffer commandBuffer, uint32_t taskCount, uint32_t firstTask) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PreCallRecordCmdDrawMeshTasksIndirectNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    uint32_t drawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void CoreChecks::PreCallRecordCmdDrawMeshTasksIndirectCountNV(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
    VkBuffer countBuffer, VkDeviceSize countBufferOffset,
    uint32_t maxDrawCount, uint32_t stride) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    UpdateStateCmdDrawType(cb_state, VK_PIPELINE_BIND_POINT_GRAPHICS);
}











void CoreChecks::PreCallRecordCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                                      uint32_t firstVertex, uint32_t firstInstance) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}



void CoreChecks::PreCallRecordCmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                                             uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}


void CoreChecks::PreCallRecordCmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                                              uint32_t stride) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}


void CoreChecks::PreCallRecordCmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                                     uint32_t count, uint32_t stride) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS);
}


void CoreChecks::PreCallRecordCmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE);
}

void CoreChecks::PreCallRecordCmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset) {
    GpuAllocateValidationResources(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE);
}

